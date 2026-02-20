import itertools

import numpy as np

from random_search_baselines.problems.problem_utils import BaseEvaluator, ProblemConfig, helper


class CirclePackingEvaluator(BaseEvaluator):
    """Evaluates circle-packing algorithms."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config['problem_parameters']
        self.n = self.params['n_circles']

    def get_helper_prelude(self, problem) -> str:
        return "import itertools"  # verify_circles uses itertools

    @staticmethod
    @helper
    def verify_circles(circles: np.ndarray) -> bool:
        """Checks that the circles are disjoint and lie inside a unit square."""
        for c1, c2 in itertools.combinations(circles, 2):
            d = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            radii_sum = c1[2] + c2[2]
            if d < radii_sum:
                return False

        for x, y, r in circles:
            if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
                return False
        return True

    def parse_output(self, raw_output, problem):
        if raw_output is None:
            raise ValueError("Output is None")

        if not isinstance(raw_output, tuple) or len(raw_output) != 3:
            raise ValueError("Invalid output format from algorithm (expected tuple of length 3)")
        centers, radii, sum_radii = raw_output
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        sum_radii = float(sum_radii)
        return centers, radii, sum_radii

    def validate_output(self, parsed_output, problem) -> None:
        centers, radii, _ = parsed_output
        self.assert_shape(centers, (self.n, 2), name="centers")
        self.assert_shape(radii, (self.n,), name="radii")
        if np.any(radii <= 0):
            raise ValueError("Non-positive radius")

        violations = self._count_violations(centers, radii)
        if violations > 0:
            raise ValueError(f"{violations} constraint violations")

    def compute_metrics(self, parsed_output, problem):
        _, _, sum_radii = parsed_output
        return {"sum_radii": float(sum_radii), "violations": 0}

    def default_failure_metrics(self, problem):
        return {"sum_radii": 0.0, "violations": 0}

    def _count_violations(self, centers: np.ndarray, radii: np.ndarray) -> int:
        """In practice uses isclose due to floating point errors, if need be can in the end either use a more precise
        checker or negligibly decrease the radii a la ShinkaEvolve."""
        count = 0

        # Boundary violations
        for (x, y), r in zip(centers, radii):
            if (
                    (x - r < 0 and not np.isclose(x, r)) or
                    (y - r < 0 and not np.isclose(y, r)) or
                    (x + r > 1 and not np.isclose(x, 1 - r)) or
                    (y + r > 1 and not np.isclose(y, 1 - r))
            ):
                count += 1

        # Overlap violations
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = np.linalg.norm(centers[i] - centers[j])
                radii_sum = radii[i] + radii[j]
                if dist < radii_sum and not np.isclose(dist, radii_sum):
                    count += 1

        return count
