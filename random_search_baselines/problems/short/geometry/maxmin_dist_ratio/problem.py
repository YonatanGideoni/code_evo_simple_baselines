from string import Template
from typing import Any

import numpy as np
import scipy as sp

from random_search_baselines.problems.problem_utils import BaseProblem, ProblemConfig


class MaxMinDistanceRatioProblem(BaseProblem):
    """Minimise (max/min pairwise distance)**2 for n points in R^d."""

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.n_points: int = int(self.params["n_points"])
        self.dim: int = int(self.params["dim"])

    def get_function_name(self) -> str:
        return "minimize_ratio"

    def get_namespace(self) -> dict[str, Any]:
        return {
            "np": np,
            "sp": sp,
            "n_points": self.n_points,
            "dim": self.dim,
        }

    def generate_instruction(self) -> str:
        base_text = super().generate_instruction()
        return Template(base_text).safe_substitute(
            n_points=self.n_points,
            dim=self.dim,
        )
