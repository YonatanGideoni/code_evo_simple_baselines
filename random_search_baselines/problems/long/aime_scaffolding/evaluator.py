from functools import partial
from typing import Any

import numpy as np

from random_search_baselines.problems.long.aime_scaffolding.cache_utils import save_eval_dataframe_csv, get_class_hash
from random_search_baselines.problems.long.aime_scaffolding.eval_utils import query_llm, create_call_limited_query_llm, \
    agent_evaluation
from random_search_baselines.problems.problem_utils import ProblemConfig, BaseClassEvaluator


class AIMEScaffoldingEvaluator(BaseClassEvaluator):
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config['problem_parameters']

    def parse_output(self, raw_output: Any, problem: Any) -> np.ndarray:
        if raw_output is None:
            raise ValueError("No output")

        return raw_output

    def validate_output(self, AgentCls, problem: Any) -> None:
        assert isinstance(AgentCls, type), "Output is not a class"
        assert hasattr(AgentCls, 'forward'), "Output does not have 'forward' method"

    def compute_metrics(self, AgentCls, problem: Any) -> dict[str, Any]:
        aime_year = self.params.get("year", 2024)
        split = self.params.get("split", "full")
        eval_model = self.params.get("eval_model")
        max_calls = self.params.get("max_calls")
        n_evals_per_q = self.params.get("n_evals_per_q", 3)
        n_eval_workers_per_scaffold = self.params.get("n_eval_workers_per_scaffold", 30)

        # Create base query_llm function
        base_query_llm = partial(query_llm, model_name=eval_model)

        # Wrap it with call limiting (max 10 calls per forward pass)
        limited_query_llm = create_call_limited_query_llm(
            base_query_llm,
            max_calls=max_calls,
        )

        accuracy, cost_total, n_problems_processed, num_llm_calls, df = agent_evaluation(
            AgentCls, limited_query_llm, year=aime_year, split=split, n_workers=n_eval_workers_per_scaffold,
            n_evals=n_evals_per_q,
        )

        eval_params = {
            "year": aime_year,
            "split": split,
            "eval_model": eval_model,
            "max_calls": max_calls,
            "n_evals": n_evals_per_q,
            "n_workers": n_eval_workers_per_scaffold,
        }
        agent_hash = get_class_hash(AgentCls, extra_params=eval_params)

        saved = save_eval_dataframe_csv(df, agent_hash)

        return {
            "accuracy": accuracy,
            "total_cost": cost_total,
            "num_problems_processed": n_problems_processed,
            "num_llm_calls": num_llm_calls,
            "agent_hash": agent_hash,
            "saved_paths": saved,
        }

    def default_failure_metrics(self, problem: Any) -> dict[str, Any]:
        return {
            'accuracy': 0.0,
            'total_cost': 0.0,
            'num_problems_processed': 0,
            'num_llm_calls': 0,
            'agent_hash': None,
            'saved_paths': None,
        }
