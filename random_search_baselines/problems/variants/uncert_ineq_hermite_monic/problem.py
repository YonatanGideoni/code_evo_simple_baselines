from string import Template

import numpy as np
from typing import Dict, Any, List
from random_search_baselines.problems.problem_utils import BaseProblem, ProblemConfig


class UncertaintyInequalityProblem(BaseProblem):
    """Defines the Uncertainty Inequality Hermite coefficients optimization problem."""

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.n_coefficients = self.params['n_coefficients']

    def get_function_name(self) -> str:
        return "find_hermite_coefficients"

    def get_namespace(self) -> Dict[str, Any]:
        return {
            'np': np,
            'List': List,
            'n_coefficients': self.n_coefficients,
            'scipy': __import__('scipy', fromlist=['special']),
        }

    def generate_instruction(self) -> str:
        base_text = super().generate_instruction()
        return Template(base_text).safe_substitute(
            n_coefficients=self.n_coefficients
        )
