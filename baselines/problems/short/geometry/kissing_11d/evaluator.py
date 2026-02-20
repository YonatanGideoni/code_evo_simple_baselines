from typing import Any, Tuple

import math
import numpy as np
import scipy as sp

from baselines.problems.problem_utils import (
    BaseEvaluator,
    ProblemConfig,
    helper,
)


class KissingNumberEvaluator(BaseEvaluator):
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]
        self.dim: int = int(self.params.get("dim", 11))

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np\nimport scipy as sp"

    # --------- helper exposed to the model ---------

    @staticmethod
    @helper
    def check_constraints(points: np.ndarray) -> bool:
        # Reject non-finite
        if not np.isfinite(points).all():
            return False

        # reject above known upper bound
        if len(points) > 868:
            return False

        # Nonzero vectors (quick guard even for int path)
        if np.any(np.linalg.norm(points, axis=1) == 0):
            return False

        if np.issubdtype(points.dtype, np.integer):
            # Exact integer path: compare squared distances/norms as Python ints
            P = points.astype(object)
            n = P.shape[0]
            max_n2 = 0
            for i in range(n):
                v = P[i]
                n2 = int(np.dot(v, v))
                if n2 == 0:
                    return False
                if n2 > max_n2:
                    max_n2 = n2
            min_d2 = None
            for i in range(n - 1):
                vi = P[i]
                for j in range(i + 1, n):
                    diff = vi - P[j]
                    d2 = int(np.dot(diff, diff))
                    if min_d2 is None or d2 < min_d2:
                        min_d2 = d2
            return min_d2 is not None and min_d2 >= max_n2

        # Floating path
        d = sp.spatial.distance.pdist(points)
        norms = np.linalg.norm(points, axis=1)
        min_d = d.min()
        max_norm = norms.max()
        return min_d >= max_norm

    # --------- internal utilities ---------

    def _exact_metrics_int(self, points: np.ndarray) -> Tuple[int, int]:
        """Return (min_pairwise_dist_sq, max_norm_sq) using exact integer arithmetic."""
        P = points.astype(object)
        n = P.shape[0]
        # max ||x||^2
        max_n2 = 0
        for i in range(n):
            n2 = int(np.dot(P[i], P[i]))
            if n2 > max_n2:
                max_n2 = n2
        # min ||x-y||^2
        min_d2 = None
        for i in range(n - 1):
            vi = P[i]
            for j in range(i + 1, n):
                diff = vi - P[j]
                d2 = int(np.dot(diff, diff))
                if (min_d2 is None) or (d2 < min_d2):
                    min_d2 = d2
        return (0 if min_d2 is None else int(min_d2), int(max_n2))

    def _float_metrics(self, points: np.ndarray) -> Tuple[float, float]:
        d = sp.spatial.distance.pdist(points)
        norms = np.linalg.norm(points, axis=1)
        return float(d.min()), float(norms.max())

    def _check_constraints(self, points: np.ndarray) -> Tuple[bool, dict]:
        if not np.isfinite(points).all():
            return False, {"reason": "Non-finite entries"}

        if len(points) > 868:
            return False, {"reason": "Exceeds known upper bound of 868"}

        if points.ndim != 2 or points.shape[1] != self.dim:
            return False, {"reason": f"Shape must be (n,{self.dim}), got {points.shape}"}
        # 0 ∉ C
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms == 0.0):
            return False, {"reason": "Zero vector present"}

        if np.issubdtype(points.dtype, np.integer):
            min_d2, max_n2 = self._exact_metrics_int(points)
            ok = (min_d2 >= max_n2)
            return ok, {
                "mode": "integer-exact",
                "min_pairwise_distance_squared": int(min_d2),
                "max_norm_squared": int(max_n2),
                "margin_squared": int(min_d2 - max_n2),
            }

        # float path
        min_d, max_norm = self._float_metrics(points)
        ok = min_d >= max_norm
        return ok, {
            "mode": "float",
            "min_pairwise_distance": float(min_d),
            "max_norm": float(max_norm),
            "margin": float(min_d - max_norm),
        }

    # --------- template hooks ---------

    def parse_output(self, raw_output: Any, problem: Any) -> np.ndarray:
        if raw_output is None:
            raise ValueError("No output")

        # Preserve dtype (int vs float) as submitted by the solver
        points = np.asarray(raw_output)
        return points

    def validate_output(self, points: np.ndarray, problem: Any) -> None:
        self.assert_all_finite(points, name="points")
        ok, info = self._check_constraints(points)
        if not ok:
            reason = info.get("reason", "Lemma constraint violated")
            # Provide helpful diagnostics if available
            details = ", ".join(f"{k}={v}" for k, v in info.items() if k != "reason")
            msg = reason if not details else f"{reason}; {details}"
            raise ValueError(msg)

        # Minimum cardinality
        if points.shape[0] < 2:
            raise ValueError("Provide at least two points (n ≥ 2).")

    def compute_metrics(self, points: np.ndarray, problem: Any) -> dict[str, Any]:
        # Mode-specific metrics
        if np.issubdtype(points.dtype, np.integer):
            min_d2, max_n2 = self._exact_metrics_int(points)
            min_d = math.sqrt(float(min_d2))
            max_norm = math.sqrt(float(max_n2))
            mode = "integer-exact"
            margin = float(min_d - max_norm)
            extra = {
                "min_pairwise_distance_squared": int(min_d2),
                "max_norm_squared": int(max_n2),
                "margin_squared": int(min_d2 - max_n2),
            }
        else:
            min_d, max_norm = self._float_metrics(points)
            mode = "float"
            margin = float(min_d - max_norm)
            extra = {}

        norms = np.linalg.norm(points.astype(float), axis=1)
        centers = (2.0 * points.astype(float) / norms[:, None]).tolist()

        return {
            "mode": mode,
            "kissing_count": int(points.shape[0]),
            "min_pairwise_distance": float(min_d),
            "max_norm": float(max_norm),
            "margin": float(margin),
            "dim": int(points.shape[1]),
            "points": points.tolist(),
            "sphere_centers": centers,
            **extra,
        }

    def default_failure_metrics(self, problem: Any) -> dict[str, Any]:
        return {
            "mode": "unknown",
            "kissing_count": 0,
            "min_pairwise_distance": 0.0,
            "max_norm": 0.0,
            "margin": float("-inf"),
            "dim": self.dim,
            "points": [],
            "sphere_centers": [],
        }
