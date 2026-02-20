from string import Template
from typing import Any, Dict

import numpy as np

from random_search_baselines.problems.problem_utils import BaseProblem, ProblemConfig


class KissingNumberProblem(BaseProblem):
    """
    B.11 — Kissing number in dimension `dimension`.

    The model must implement a function `kissing_points()` that returns an
    array-like of shape (n, dimension) representing a set C with 0 ∉ C and

        min_{x≠y in C} ||x - y||_2  ≥  max_{x in C} ||x||_2.

    This certifies a valid kissing configuration of size |C| via the lemma.
    We accept either:
      • integer coordinates (checked EXACTLY using Python big integers), or
      • float coordinates (checked with a tiny tolerance).

    Helpers exposed in the namespace:
      - check_lemma_constraints(points, tol=None) -> bool
      - dimension (int)
      - np
    """

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.dimension: int = int(self.params.get("dimension", 11))

    # The function name your agent must define.
    def get_function_name(self) -> str:
        return "kissing_points"

    # Variables / helpers available to the agent when executing its code.
    def get_namespace(self) -> Dict[str, Any]:
        d = self.dimension

        return {
            "np": np,
            "dimension": d,
        }

    def generate_instruction(self) -> str:
        base_text = super().generate_instruction()
        return Template(base_text).safe_substitute(
            dim=self.dimension,
        )
