from string import Template
from typing import Any

import numpy as np

from baselines.problems.problem_utils import BaseProblem, ProblemConfig


class HeilbronnTrianglesProblem(BaseProblem):
    """Heilbronn problem: maximise the smallest triangle area formed by n points inside a fixed equilateral triangle."""

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.n_points: int = int(self.params["n_points"])
        # Fixed equilateral triangle with side length 1
        self.triangle_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]], dtype=float)

    def get_function_name(self) -> str:
        return "heilbronn_points"

    def get_namespace(self) -> dict[str, Any]:
        return {
            "np": np,
            "n_points": self.n_points,
            "triangle_vertices": self.triangle_vertices,
        }

    # Use base get_instruction (substitutes ${max_execution_time}), then fill remaining placeholders.
    def generate_instruction(self) -> str:
        base_text = super().generate_instruction()
        return Template(base_text).safe_substitute(n_points=self.n_points)
