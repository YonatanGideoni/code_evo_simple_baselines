import concurrent.futures
import json
import logging
import time
from dataclasses import asdict
from typing import Any

from baselines.archive_utils import ProblemWithArchive
from baselines.config_utils import get_base_parser, merge_and_override_configs, load_api_config
from baselines.gemini_utils import GeminiAlgorithmGenerator
from baselines.pricing_utils import _cost_from_usage
from baselines.problems.problem_utils import ProblemLoader

# Configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _evaluate_one(code_and_index: tuple, problem, evaluator):
    """Helper executed in a worker process."""
    idx, code = code_and_index
    result = evaluator.evaluate_algorithm(code, problem)
    return idx, result


def _usage_to_counts(u: dict) -> dict:
    pt = int(u.get("prompt_tokens", 0))
    ct = int(u.get("candidates_tokens", 0))
    tt = u.get("thoughts_tokens", 0)
    tt = tt if tt is not None else 0
    cached = u['cached_prompt_tokens']
    cached = cached if cached is not None else 0
    out_tokens = ct + tt
    return {
        "prompt_tokens": pt,
        "output_tokens": out_tokens,
        "cached_prompt_tokens": cached,
        "total_tokens": pt + out_tokens
    }


def apply_cli_overrides(api_params: dict, args) -> dict:
    """
    Merge CLI-specified overrides into api_params.
    Only sets keys for which the user provided a value on the CLI.
    """
    api_params = (api_params or {}).copy()

    if getattr(args, "max_retries", None) is not None:
        api_params["max_retries"] = int(args.max_retries)
    if getattr(args, "model", None) is not None:
        api_params["gemini_model"] = args.model
    if getattr(args, "eval_workers", None) is not None:
        api_params["num_workers"] = int(args.eval_workers)
    if getattr(args, "gen_workers", None) is not None:
        api_params["max_concurrency"] = int(args.gen_workers)

    return api_params


def resolve_config(args):
    _, _, prob_config = ProblemLoader.load_problem(args.problem)

    api_config_path = args.api_config

    api_config = load_api_config(api_config_path)

    config_dict, meta_conf = merge_and_override_configs(
        prob_config=prob_config,
        api_config=api_config,
        args=args,
    )

    return config_dict, meta_conf


def gen_algs(args, problem, config, meta_conf):
    api_key = args.api_key
    num_algorithms = args.num_algorithms

    if args.archive_path is not None:
        # TODO add params, maybe from some config, to the archive - e.g. how many problems to pick (k), selection strategy, etc.
        problem = ProblemWithArchive(problem, args.archive_path, args.n_examples_from_archive)

    # Initialize Gemini generator
    model_name = meta_conf["model_name"]
    print(f"Initializing Gemini model: {model_name}")
    generator = GeminiAlgorithmGenerator(api_key=api_key, config=config)

    if args.max_exec_time is not None:
        # monkey patch problem.max_execution_time
        problem.params['max_execution_time'] = float(args.max_exec_time)

    # Generate algorithms
    start_time = time.time()
    algorithms = generator.generate_algorithms(problem, num_algorithms)
    generation_time = time.time() - start_time

    usage = getattr(generator, 'last_usage', None)
    generated_completions = getattr(generator, 'generated_completions', None)
    prompts = getattr(generator, 'prompts_used', None)

    if len(algorithms) < num_algorithms:
        logger.warning("Only %d algorithms produced; continuing anyway", len(algorithms))

    print(f"Generation completed in {generation_time:.2f} seconds")

    return {
        'algorithms': algorithms,
        'generation_time': generation_time,
        'usage': usage,
        'generated_completions': generated_completions,
        'prompts': prompts,
    }


def run_evaluation_gemini(args):
    start_time = time.time()

    problem_name = args.problem

    # Load problem config
    problem, evaluator, prob_config = ProblemLoader.load_problem(problem_name)

    config_dict, meta_conf = resolve_config(args)

    gen_res = gen_algs(args, problem, config_dict, meta_conf)

    algorithms = gen_res['algorithms']
    model_name = meta_conf["model_name"]
    generation_time = gen_res['generation_time']
    if not algorithms:
        print("No algorithms generated successfully")
        evaluation_summary = {
            'problem_name': problem_name,
            'model_name': model_name,
            'total_algorithms': 0,
            'successful_algorithms': 0,
            'success_rate': 0.0,
            'generation_time': generation_time,
            'eval_time': 0.0,
            'total_time': generation_time,
            'tokens': {
                'input': 0,
                'output': 0,
                'cached_prompt_tokens': 0,
                'total': 0
            },
            'cost_usd': {
                'input': 0.0,
                'output': 0.0,
                'cached_prompt': 0.0,
                'total': 0.0,
            },
            'config': config_dict,
        }
        return {'metadata': evaluation_summary, 'results': []}

    # Evaluate algorithms
    num_workers = int(config_dict["api_params"].get("num_workers", 8))
    print(f"Evaluating {len(algorithms)} algorithms on {num_workers} workers...")

    start_eval = time.time()
    results = [None] * len(algorithms)  # pre-allocate to retain order

    def _zero_counts():
        return {'prompt_tokens': 0, 'output_tokens': 0, 'cached_prompt_tokens': 0}

    usage = gen_res['usage']
    generated_completions = gen_res['generated_completions']
    prompts = gen_res['prompts']
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_to_idx = {
            pool.submit(_evaluate_one, (i, code), problem, evaluator): i
            for i, code in enumerate(algorithms)
        }

        for j, future in enumerate(concurrent.futures.as_completed(future_to_idx), 1):
            if j % 5 == 0 or j == len(algorithms):
                print(f"Evaluated {j}/{len(algorithms)}")
            idx, result = future.result()
            results[idx] = asdict(result)
            results[idx]['algorithm_id'] = idx + 1
            results[idx]['algorithm_code'] = algorithms[idx]
            results[idx]['full_completion'] = generated_completions[idx]
            results[idx]['prompt_used'] = prompts[idx]

            # Per-index usage if available
            u = usage[idx] if idx < len(usage) else None
            counts = _usage_to_counts(u) if u is not None else _zero_counts()
            for k in ('prompt_tokens', 'output_tokens', 'cached_prompt_tokens'):
                counts.setdefault(k, 0)
            results[idx]['token_usage'] = counts

            per_call_cost = _cost_from_usage(model_name=model_name, usage_counts=counts)
            results[idx]['api_cost_usd'] = per_call_cost

    eval_time = time.time() - start_eval

    successful_results = [r for r in results if r and r.get("success")]
    if not successful_results:
        print("\nAll algorithms failed! (saving results anyway)")

    sum_in = sum(r["token_usage"]["prompt_tokens"] for r in results if r)
    sum_out = sum(r["token_usage"]["output_tokens"] for r in results if r)
    sum_cached = sum(r["token_usage"]["cached_prompt_tokens"] for r in results if r)
    sum_total = sum_in + sum_out

    print(f"\n=== TOKEN USAGE SUMMARY ===")
    print(f"Input tokens:  {sum_in}  (cached hits: {sum_cached})")
    print(f"Output tokens: {sum_out}")
    print(f"Total tokens:  {sum_total}")

    agg_input_usd = 0.0
    agg_output_usd = 0.0
    agg_cache_usd = 0.0
    agg_total_usd = 0.0

    for r in results:
        if not r:
            continue
        c = r.get("api_cost_usd") or {}
        agg_input_usd += float(c.get("input_usd", 0.0))
        agg_output_usd += float(c.get("output_usd", 0.0))
        agg_cache_usd += float(c.get("cache_usd", 0.0))
        agg_total_usd += float(c.get("total_usd", 0.0))

    print(f"\n=== COST SUMMARY (USD, Standard / non-batch) ===")
    print(f"Model: {model_name}")
    print(f"Input cost:   ${agg_input_usd:,.4f}")
    if agg_cache_usd > 0:
        print(f"Cache usage:  ${agg_cache_usd:,.4f}   (context caching usage only; excludes storage/hour)")
    print(f"Output cost:  ${agg_output_usd:,.4f}")
    print(f"---------------------------------------")
    print(f"TOTAL API COST: ${agg_total_usd:,.4f}")

    evaluation_summary = {
        'problem_name': problem_name,
        'model_name': model_name,
        'total_algorithms': len(algorithms),
        'successful_algorithms': len(successful_results),
        'success_rate': len(successful_results) / len(algorithms),
        'generation_time': generation_time,
        'eval_time': eval_time,
        'total_time': time.time() - start_time,
        'tokens': {
            'input': sum_in,
            'output': sum_out,
            'cached_prompt_tokens': sum_cached,
            'total': sum_total
        },
        'cost_usd': {
            'input': agg_input_usd,
            'output': agg_output_usd,
            'cached_prompt': agg_cache_usd,
            'total': agg_total_usd,
        },
        'config': config_dict,
        'meta_config': meta_conf,
    }

    print(f"\n=== SUMMARY ===")
    print(f"Success Rate: {evaluation_summary['success_rate']:.1%}")
    print(f"Generation Time: {evaluation_summary['generation_time']:.2f}s")
    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"Total Time: {evaluation_summary['total_time']:.2f}s")

    return {
        'metadata': evaluation_summary,
        'results': results
    }


def save_results(results_data: dict[str, Any], filename: str):
    """Save results to a JSON file."""
    if results_data is not None:
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {filename}")
    else:
        print("No results to save")


def gen_solutions_for_problem(args):
    results = run_evaluation_gemini(args)

    res_path = args.save_results
    if res_path is None:
        resolved_model = None
        if results and isinstance(results, dict):
            resolved_model = results.get('metadata', {}).get('model_name')
        if not resolved_model:
            resolved_model = args.model if args.model else "default_model"
        n_algs = args.num_algorithms
        use_archive = args.archive_path is not None
        res_path = f"results_{args.problem}_{n_algs}_{resolved_model}_archive{use_archive}.json"

    if results:
        save_results(results, res_path)


def main():
    parser = get_base_parser()
    parser.add_argument("--list-problems", action="store_true", help="List available problems")
    args = parser.parse_args()

    if args.list_problems:
        problems = ProblemLoader.list_problems()
        if problems:
            print("Available problems:")
            for p in problems:
                print(f"  - {p}")
        else:
            print("No problems found.")
        return

    if args.problem is None:
        print("Error: --problem argument is required unless --list-problems is specified.")
        return

    gen_solutions_for_problem(args)


if __name__ == "__main__":
    main()
