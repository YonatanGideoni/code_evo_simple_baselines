import inspect
import itertools
import textwrap

import numpy as np
from scipy.optimize import linprog

from baselines.problems.problem_utils import BaseEvaluator, ProblemConfig, helper


def run_lp(centers: np.ndarray) -> np.ndarray:
    """
    Given circle centers (shape (n,2)), solve the LP that maximizes the sum of radii
    subject to non-overlapping and staying inside the unit square constraints.
    """
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (n, 2)")

    n = centers.shape[0]
    # Objective: maximize sum r_i  <=>  minimize -sum r_i
    c = -np.ones(n, dtype=float)

    # Pairwise non-overlap constraints: for each i<j, r_i + r_j <= d_ij
    A_ub_rows = []
    b_ub = []

    for i, j in itertools.combinations(range(n), 2):
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        d_ij = np.hypot(dx, dy)  # Euclidean distance
        row = np.zeros(n, dtype=float)
        row[i] = 1.0
        row[j] = 1.0
        A_ub_rows.append(row)
        b_ub.append(d_ij)

    A_ub = np.vstack(A_ub_rows)
    b_ub = np.array(b_ub, dtype=float)

    # Bounds: 0 <= r_i <= distance to nearest boundary
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r_to_edge = min(x, 1.0 - x, y, 1.0 - y)
        if max_r_to_edge < 0:
            raise ValueError(f"Center {i} = ({x:.6g},{y:.6g}) is outside the unit square")
        bounds.append((0.0, float(max_r_to_edge)))

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    radii = res.x
    radii = np.maximum(radii, 1e-10)  # Ensure positivity, can happen due to numerical errors
    return radii


class CirclePackingEvaluator(BaseEvaluator):
    """Evaluates circle-packing algorithms."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config['problem_parameters']
        self.n = self.params['n_circles']

    def get_helper_prelude(self, problem) -> str:
        prelude = "import itertools\nimport numpy as np\nfrom scipy.optimize import linprog\n"

        # grab the module-level run_lp source
        run_lp_src = inspect.getsource(run_lp)
        run_lp_src = textwrap.dedent(run_lp_src)

        return prelude + "\n\n" + run_lp_src

    @staticmethod
    @helper
    def find_max_sum_radii(centers: np.ndarray) -> float:
        radii = run_lp(centers)
        return float(np.sum(radii))

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

        if len(raw_output) != 26 or len(raw_output[0]) != 2:
            raise ValueError("Invalid output format from algorithm (expected np ndarray of shape (26,2) for centers)")
        centers = raw_output
        centers = np.asarray(centers, dtype=float)
        radii = run_lp(centers)
        sum_radii = float(sum(radii))
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
