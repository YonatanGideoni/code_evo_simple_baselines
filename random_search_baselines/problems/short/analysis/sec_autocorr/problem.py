from typing import Dict, Any, List

import numpy as np

from random_search_baselines.problems.problem_utils import BaseProblem


class SecondAutocorrProblem(BaseProblem):
    def get_function_name(self) -> str:
        return "find_step_heights"

    def get_namespace(self) -> Dict[str, Any]:
        return {
            "np": np,
            "List": List,
            "max_execution_time": self.params.get("max_execution_time", 60.0),
        }
