from typing import Any

import numpy as np
import scipy as sp

from random_search_baselines.problems.problem_utils import (
    BaseEvaluator,
    ProblemConfig,
    helper,
)


class MaxMinDistanceRatioEvaluator(BaseEvaluator):
    """Evaluates (max/min pairwise distance)**2 from returned point sets."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]
        self.n_points: int = int(self.params["n_points"])
        self.dim: int = int(self.params["dim"])

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np\nimport scipy as sp"

    @staticmethod
    @helper
    def compute_ratio_squared(points: np.ndarray) -> float:
        # Assume valid input; use SciPy for pairwise distances.
        d = sp.spatial.distance.pdist(points)
        return (d.max() / d.min()) ** 2

    # --------- template hooks ---------

    def parse_output(self, raw_output: Any, problem: Any) -> np.ndarray:
        points = np.asarray(raw_output, dtype=float)
        return points

    def validate_output(self, points: np.ndarray, problem: Any) -> None:
        self.assert_shape(points, (self.n_points, self.dim), name="points")
        self.assert_all_finite(points, name="points")
        d = sp.spatial.distance.pdist(points)
        if d.size == 0 or d.min() <= 0.0:
            raise ValueError("Minimum pairwise distance must be positive.")

    def compute_metrics(self, points: np.ndarray, problem: Any) -> dict[str, Any]:
        d = sp.spatial.distance.pdist(points)
        min_d = d.min()
        max_d = d.max()
        ratio_squared = (max_d / min_d) ** 2
        return {
            "ratio_squared": float(ratio_squared),
            "min_distance": float(min_d),
            "max_distance": float(max_d),
            "n_points": int(points.shape[0]),
            "dim": int(points.shape[1]),
            "points": points.tolist(),
        }

    def default_failure_metrics(self, problem: Any) -> dict[str, Any]:
        return {
            "ratio_squared": float("inf"),
            "min_distance": 0.0,
            "max_distance": 0.0,
            "n_points": self.n_points,
            "dim": self.dim,
            "points": [],
        }
