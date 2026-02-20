import concurrent.futures
import json
import multiprocessing
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Any, Optional

import numpy as np
import pandas as pd

from baselines.problems.long.aime_scaffolding.cache_utils import get_class_hash
from baselines.problems.long.aime_scaffolding.eval_utils import (
    query_llm,
    create_call_limited_query_llm,
    evaluate_math_correctness,
    is_equiv,
)


class Agent:
    def __init__(
            self,
            query_llm: Callable,
            temperature: float = 0.7,
            num_samples: int = 15,
    ):
        self.output_format_instructions = (
            "On the final line output only the digits of the answer (0-999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{...} command."
        )
        self.query_llm = query_llm
        self.temperature = temperature
        self.num_samples = num_samples

    def forward(self, problem: str) -> tuple[list[str], float]:
        system_prompt, task_prompt = self.get_prompt_for_task(problem)

        responses: list[str] = []
        total_cost = 0.0

        for _ in range(self.num_samples):
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            responses.append(response)
            total_cost += cost

        return responses, total_cost

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = (
            "You are a skilled mathematician who is an expert in thinking step-by-step to reach a solution."
        )
        task_prompt = (
            f"Solve the following math problem: {problem}\n\n"
            f"Let's think step by step.\n\n"
            f"{self.output_format_instructions}"
        )
        return system_prompt, task_prompt


def majority_vote(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0]


def pick_representative_response(
        responses: list[str],
        extracted_answers: list[str],
        majority_answer: str,
) -> str:
    """
    Returns the first response whose extracted answer matches the majority.
    If none match (should be rare), returns the first response.
    """
    for resp, ans in zip(responses, extracted_answers):
        if ans == majority_answer:
            return resp
    return responses[0] if responses else ""


def process_example(idx: int, example: pd.Series, agent: Agent, query_llm: Callable):
    # Reset call count for each example if using call-limited query_llm
    if hasattr(query_llm, "reset_calls"):
        query_llm.reset_calls()

    problem = str(example["problem"]).strip()
    solution = example["answer"]

    responses, cost = agent.forward(problem)

    # IMPORTANT: includes malformed answers
    # We extract via evaluate_math_correctness, which returns llm_answer="" when parsing fails.
    extracted_answers: list[str] = []
    for r in responses:
        llm_answer, _true_answer, _correct = evaluate_math_correctness(r, solution)
        extracted_answers.append(llm_answer)

    majority_answer = majority_vote(extracted_answers) if extracted_answers else ""

    final_response = pick_representative_response(responses, extracted_answers, majority_answer)
    llm_answer, true_answer, correct = evaluate_math_correctness(final_response, solution)

    num_llm_calls = query_llm.get_call_count() if hasattr(query_llm, "get_call_count") else None

    return {
        "id": idx,
        "problem": problem,
        "true_answer": str(true_answer),
        "answers": extracted_answers,  # list[str]
        "majority_answer": majority_answer,
        "llm_answer": llm_answer,
        "correct": bool(correct),
        "cost": float(cost),
        "num_llm_calls": int(num_llm_calls) if num_llm_calls is not None else None,
        "response": final_response,
    }


def agent_evaluation(
        AgentCls,
        query_llm: Callable,
        year: int = 2024,
        split: str = "full",
        n_workers: int = 30,
        agent_kwargs: Optional[dict] = None,
) -> tuple[float, float, int, int, pd.DataFrame]:
    agent_kwargs = agent_kwargs or {}

    base_dir = Path(__file__).resolve().parent
    math_test_set = pd.read_csv(base_dir / "AIME_Dataset_1983_2025.csv")
    math_test_set = math_test_set[math_test_set["Year"] == year]
    if split != "full":
        if split == "odd":
            math_test_set = math_test_set.iloc[::2]
        elif split == "even":
            math_test_set = math_test_set.iloc[1::2]
        else:
            raise ValueError(f"Unknown split: {split}")

    agent = AgentCls(query_llm, **agent_kwargs)

    results = []
    max_workers = min(n_workers, multiprocessing.cpu_count())

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_example, i, example, agent, query_llm): i
            for i, (_, example) in enumerate(math_test_set.iterrows())
        }

        total, correct_count, total_llm_calls, cost_total = 0, 0, 0, 0.0

        for future in concurrent.futures.as_completed(future_to_idx):
            total += 1
            row = future.result()
            results.append(row)

            cost_total += row["cost"]
            if row["correct"]:
                correct_count += 1
            if row["num_llm_calls"] is not None:
                total_llm_calls += row["num_llm_calls"]

            if total % 10 == 0 or total == len(math_test_set):
                acc = 100.0 * correct_count / max(total, 1)
                avg_calls = total_llm_calls / max(total, 1) if total_llm_calls else 0.0
                print(
                    f"[{total}/{len(math_test_set)}] "
                    f"Acc={acc:.2f}%  Cost={cost_total:.4f}  "
                    f"Calls={total_llm_calls}  AvgCalls={avg_calls:.2f}"
                )

    if total == 0:
        raise ValueError("No examples were processed.")

    final_accuracy = 100.0 * correct_count / total
    elapsed = time.time() - start_time
    print(f"Done. Final accuracy: {final_accuracy:.2f}% | Cost: {cost_total:.4f} | Time: {elapsed:.2f}s")

    df = pd.DataFrame(results)
    return final_accuracy, cost_total, total, total_llm_calls, df


@dataclass
class BootstrapConfig:
    k_min: int = 1
    k_max: int = 10
    n_boot: int = 1000
    seed: int = 0
    ci_quantiles: tuple[float, float] = (0.05, 0.95)


def compute_bootstrapped_majority_at_k(
        df: pd.DataFrame,
        cfg: BootstrapConfig = BootstrapConfig(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    pools: list[list[str]] = []
    truths: list[str] = []

    for _, row in df.iterrows():
        pool = row["answers"]
        if not isinstance(pool, list):
            pool = [] if pool is None else list(pool)
        # Keep malformed answers as "" (do not drop)
        pools.append([a if isinstance(a, str) else "" for a in pool])
        truths.append(str(row["true_answer"]))

    n = len(pools)
    if n == 0:
        raise ValueError("Empty df; cannot compute majority@k.")

    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        accs = np.empty(cfg.n_boot, dtype=float)

        for b in range(cfg.n_boot):
            correct = 0

            for pool, truth in zip(pools, truths):
                if not pool:
                    pred = ""
                else:
                    kk = min(k, len(pool))
                    # subsample answers WITHOUT replacement (as requested)
                    subs = rng.choice(pool, size=kk, replace=False).tolist()
                    pred = majority_vote(subs) if subs else ""

                if is_equiv(pred, truth):
                    correct += 1

            accs[b] = 100.0 * correct / n

        mean = float(accs.mean())
        std = float(accs.std(ddof=1))
        lo, hi = np.quantile(accs, list(cfg.ci_quantiles)).tolist()

        rows.append(
            {
                "k": int(k),
                "mean_accuracy": mean,
                "std": std,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "ci_quantiles": f"{cfg.ci_quantiles[0]:.2f},{cfg.ci_quantiles[1]:.2f}",
                "n_boot": int(cfg.n_boot),
                "n_questions": int(n),
            }
        )

    return pd.DataFrame(rows)


def print_metrics(metrics_df: pd.DataFrame) -> None:
    print("\nBootstrapped majority@k accuracies (mean with CI):")
    for _, r in metrics_df.iterrows():
        print(
            f"  k={int(r['k']):2d}: "
            f"{r['mean_accuracy']:.2f}% "
            f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] "
            f"(std={r['std']:.2f}, boot={int(r['n_boot'])}, q={r['ci_quantiles']})"
        )


def save_problems_and_answers_csv(df: pd.DataFrame, out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()

    def _serialize_answers(x):
        # None / NaN -> []
        if x is None:
            return "[]"
        try:
            # pandas may store missing as float NaN
            if isinstance(x, float) and pd.isna(x):
                return "[]"
        except Exception:
            pass

        # already a string (maybe already JSON) -> keep
        if isinstance(x, str):
            return x

        # list -> json
        if isinstance(x, list):
            return json.dumps(x, ensure_ascii=False)

        # other iterables -> list -> json
        try:
            return json.dumps(list(x), ensure_ascii=False)
        except Exception:
            # worst-case fallback
            return json.dumps([str(x)], ensure_ascii=False)

    if "answers" in df_out.columns:
        df_out["answers"] = df_out["answers"].apply(_serialize_answers)

    df_out.to_csv(out_csv, index=False)
    return out_csv


def save_pickle(obj: Any, out_pkl: Path) -> Path:
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(obj, f)
    return out_pkl


def main():
    aime_year = 2024
    split = "full"
    eval_model = "gpt-4.1-nano"

    # You want 15 samples/question
    num_samples = 15
    temperature = 0.7

    # Must allow >= 15 calls per forward
    max_calls = num_samples

    n_eval_workers_per_scaffold = 30

    # Create base query_llm
    base_query = partial(query_llm, model_name=eval_model)

    # Wrap with call limiting
    limited_query = create_call_limited_query_llm(base_query, max_calls=max_calls)

    eval_params = {
        "year": aime_year,
        'split': split,
        "eval_model": eval_model,
        "max_calls": max_calls,
        "n_workers": n_eval_workers_per_scaffold,
        "num_samples": num_samples,
        "temperature": temperature,
    }
    agent_hash = get_class_hash(Agent, extra_params=eval_params)

    accuracy, cost_total, n_processed, num_llm_calls, df = agent_evaluation(
        Agent,
        limited_query,
        year=aime_year,
        split=split,
        n_workers=n_eval_workers_per_scaffold,
        agent_kwargs={"temperature": temperature, "num_samples": num_samples},
    )

    out_dir = Path("eval_outputs_2024")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure filenames include YEAR (as requested)
    problems_answers_csv = out_dir / f"{aime_year}_{agent_hash}_problems_and_answers.csv"
    saved_csv = save_problems_and_answers_csv(df, problems_answers_csv)

    # Bootstrapped majority@k for k=1..10 (subsampling without replacement)
    boot_cfg = BootstrapConfig(k_min=1, k_max=10, n_boot=1000, seed=0, ci_quantiles=(0.025, 0.975))
    metrics_df = compute_bootstrapped_majority_at_k(df, boot_cfg)

    metrics_csv = out_dir / f"{aime_year}_{agent_hash}_majority_at_k_bootstrap.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print_metrics(metrics_df)

    # Pickle extra metadata
    extras = {
        "agent_hash": agent_hash,
        "eval_params": eval_params,
        "final_accuracy_percent": accuracy,
        "cost_total": cost_total,
        "n_processed": n_processed,
        "num_llm_calls": num_llm_calls,
        "bootstrap_config": boot_cfg,
        "bootstrap_metrics_df": metrics_df,
        "saved_csv": str(saved_csv),
        "saved_metrics_csv": str(metrics_csv),
    }
    extras_pkl = out_dir / f"{aime_year}_{agent_hash}_extras.pkl"
    saved_pkl = save_pickle(extras, extras_pkl)

    print(
        f"\nSummary\n"
        f"- Accuracy (agent representative response): {accuracy:.2f}%\n"
        f"- Total cost: {cost_total:.4f}\n"
        f"- Problems processed: {n_processed}\n"
        f"- LLM calls: {num_llm_calls}\n"
        f"- Saved problems+answers CSV: {saved_csv}\n"
        f"- Saved bootstrap metrics CSV: {metrics_csv}\n"
        f"- Saved extras pickle: {saved_pkl}\n"
    )


if __name__ == "__main__":
    main()
