import itertools
from typing import Any

import numpy as np

from baselines.problems.problem_utils import (
    BaseEvaluator,
    ProblemConfig,
    helper,
)


class HeilbronnTrianglesEvaluator(BaseEvaluator):
    """Evaluates the normalised minimum triangle area for returned point sets."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]
        self.n_points: int = int(self.params["n_points"])

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np\nimport itertools"

    # ---------- helpers exposed to the algorithm (available in its namespace) ----------

    @staticmethod
    @helper
    def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0

    @staticmethod
    @helper
    def compute_min_area(points: np.ndarray) -> float:
        base_area = np.sqrt(3.0) / 4.0  # area of equilateral triangle with side length 1
        min_area = float("inf")
        for i, j, k in itertools.combinations(range(points.shape[0]), 3):
            a, b, c = points[i], points[j], points[k]
            area = abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
            if area < min_area:
                min_area = area
        return min_area / base_area

    @staticmethod
    @helper
    def check_inside_triangle(points: np.ndarray) -> bool:
        rt3 = np.sqrt(3.0)
        for x, y in points:
            if not (y >= 0 and y <= rt3 * x and rt3 * x <= rt3 - y):
                return False
        return True

    # ---------- template hooks ----------

    def parse_output(self, raw_output: Any, problem: Any) -> np.ndarray:
        if raw_output is None:
            raise ValueError("Output is None")

        points = np.asarray(raw_output, dtype=float)
        return points

    def validate_output(self, points: np.ndarray, problem: Any) -> None:
        self.assert_shape(points, (self.n_points, 2), name="points")
        self.assert_all_finite(points, name="points")
        # Inside equilateral triangle inequalities
        rt3 = np.sqrt(3.0)
        for x, y in points:
            if (
                    (y < 0 and not np.isclose(y, 0)) or
                    (y > rt3 * x and not np.isclose(y, rt3 * x)) or
                    (rt3 * x > rt3 - y and not np.isclose(rt3 * x, rt3 - y))
            ):
                raise ValueError("All points must lie on or inside the specified equilateral triangle.")

    def compute_metrics(self, points: np.ndarray, problem: Any) -> dict[str, Any]:
        # Compute normalised minimum triangle area
        base_area = np.sqrt(3.0) / 4.0
        min_area = float("inf")
        for i, j, k in itertools.combinations(range(points.shape[0]), 3):
            a, b, c = points[i], points[j], points[k]
            area = abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
            if area < min_area:
                min_area = area
        min_area_normalised = (min_area / base_area) if np.isfinite(min_area) else 0.0
        return {
            "min_area": float(min_area_normalised),
            "n_points": int(points.shape[0]),
            "points": points.tolist(),
        }

    def default_failure_metrics(self, problem: Any) -> dict[str, Any]:
        return {
            "min_area": 0.0,
            "n_points": self.n_points,
            "points": [],
        }
