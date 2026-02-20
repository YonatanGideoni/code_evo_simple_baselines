import numpy as np
from typing import Dict, Any, List
from baselines.problems.problem_utils import BaseProblem, ProblemConfig


class SumsAndDifferencesProblem(BaseProblem):
    """
    Construction problem for AlphaEvolve B.6: "Sums and differences of finite sets".

    The target function must return a 1-D *sorted* NumPy array of non‑negative
    integers U (dtype=int) with unique entries and with 0 ∈ U. No artificial size caps;
    the runtime limit governs.
    """

    def __init__(self, config: ProblemConfig):
        super().__init__(config)

    def get_function_name(self) -> str:
        return "propose_integer_set"

    def get_namespace(self) -> Dict[str, Any]:
        return {
            "np": np,
            "List": List,
            "max_execution_time": self.params.get("max_execution_time", 60.0),
        }
