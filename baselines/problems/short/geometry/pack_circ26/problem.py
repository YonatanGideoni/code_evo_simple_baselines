import numpy as np
from typing import Dict, Any, Tuple

from baselines.problems.problem_utils import BaseProblem, ProblemConfig


class CirclePackingProblem(BaseProblem):
    """Defines the 26-circle packing optimization problem."""

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.n = self.params['n_circles']

    def get_function_name(self) -> str:
        return "pack_circles"

    def get_namespace(self) -> Dict[str, Any]:
        return {
            'np': np,
            'Tuple': Tuple,
            'n_circles': self.n,
            'unit_square': self.params['unit_square']
        }
