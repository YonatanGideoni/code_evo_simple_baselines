import json
import os
import time

import numpy as np

from random_search_baselines.config_utils import get_base_parser
from random_search_baselines.problems.problem_utils import ProblemLoader
from random_search_baselines.solve_prob import gen_solutions_for_problem


def save_run_metadata(args, base_save_dir: str) -> None:
    metadata = {
        'problem': args.problem,
        'n_gens': args.n_gens,
        'n_algs_per_gen': args.num_algorithms,
        'selection_strategy': args.selection_strategy,
        'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'base_save_dir': base_save_dir,
    }

    os.makedirs(base_save_dir, exist_ok=True)
    with open(os.path.join(base_save_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def get_archive_path_for_gen(base_save_dir: str, gen: int) -> str:
    return os.path.join(base_save_dir, str(gen), "res.json")


def run_generation(args, base_save_dir: str, gen: int) -> None:
    archive_path = get_archive_path_for_gen(base_save_dir, gen)
    args.save_results = archive_path
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)

    if gen == 0:
        args.archive_path = None
    else:
        args.archive_path = get_archive_path_for_gen(base_save_dir, gen - 1)

    gen_solutions_for_problem(args)


def calc_gen_stats(args, base_save_dir: str, gen: int) -> dict:
    archive_path = get_archive_path_for_gen(base_save_dir, gen)
    with open(archive_path, "r") as f:
        results = json.load(f)

    metadata = results['metadata']
    success_rate = metadata["success_rate"]
    total_cost = metadata["cost_usd"]["total"]

    _, _, prob_config = ProblemLoader.load_problem(args.problem)
    score_metric_name = prob_config['conf_metric_name']
    good_answers = [r for r in results['results'] if r['success']]
    scores = [r['metrics'][score_metric_name] for r in good_answers]
    if not scores:
        scores = [0, 0]

    stats = {
        'success_rate': success_rate,
        'cost': total_cost,
        'min_score': min(scores),
        'max_score': max(scores),
        'avg_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'median_score': float(np.median(scores)),
    }

    return stats


def run_gen(args, base_save_dir, gen: int) -> dict:
    start_time = time.time()

    run_generation(args, base_save_dir, gen)

    end_time = time.time()
    gen_time = end_time - start_time

    stats = calc_gen_stats(args, base_save_dir, gen)
    stats['runtime'] = gen_time

    return stats


def backup_archive(base_save_dir, gen):
    archive_path = get_archive_path_for_gen(base_save_dir, gen)
    backup_path = archive_path + f".backup{int(time.time())}"
    os.rename(archive_path, backup_path)
    print(f"Backed up archive to {backup_path}")


def main():
    parser = get_base_parser()

    parser.add_argument('--n_examples_from_archive', type=int, default=3)
    parser.add_argument("--n_gens", type=int, default=10, help="Number of generations to run.")
    parser.add_argument("--selection_strategy", type=str, default="random", choices=["random", "best"],
                        help="Strategy for selecting algorithms from archive.")

    args = parser.parse_args()

    base_save_dir = args.save_results
    if base_save_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_save_dir = os.path.join("results", args.problem, timestamp)

    save_run_metadata(args, base_save_dir)
    time_per_gen = []
    stats_per_gen = []
    n_retries = 0
    for gen in range(args.n_gens):
        print(f"Running generation {gen + 1}/{args.n_gens}")

        gen_succeeded = False

        while not gen_succeeded:
            gen_stats = run_gen(args, base_save_dir, gen)
            if gen_stats['success_rate'] > 0:
                gen_succeeded = True
                n_retries = 0
            else:
                print(f"Generation {gen + 1} had 0% success rate. Retrying...")
                backup_archive(base_save_dir, gen)
                n_retries += 1

                if n_retries == 5:
                    print("Maximum retries reached. Exiting.")
                    gen_stats = {'success_rate': 0, 'cost': 0, 'runtime': 0}
                    break

        stats_per_gen.append(gen_stats)
        gen_time = gen_stats['runtime']
        time_per_gen.append(gen_time)

        print(f"Generation {gen + 1} completed in {gen_time:.2f} seconds.")

        if n_retries == 5:
            break

    with open(os.path.join(base_save_dir, "generation_stats.json"), "w") as f:
        json.dump(stats_per_gen, f, indent=2)


if __name__ == '__main__':
    main()
