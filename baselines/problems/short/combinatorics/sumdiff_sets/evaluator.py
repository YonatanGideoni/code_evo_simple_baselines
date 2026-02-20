import numpy as np

from baselines.problems.problem_utils import BaseEvaluator, ProblemConfig, helper


class SumsAndDifferencesEvaluator(BaseEvaluator):
    """Evaluates sums-and-differences set constructors (AlphaEvolve B.6)."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]

    def get_helper_prelude(self, problem) -> str:
        # Helper(s) below rely on NumPy
        return "import numpy as np"

    # ---------------- helpers ----------------

    @staticmethod
    @helper
    def compute_lower_bound(u: list[int]) -> float:
        """Returns the lower bound obtained from the input set u, which must satisfy min(u) == 0."""
        if min(u) != 0:
            raise AssertionError(
                "Set U must be nonnegative and must contain 0; got minimum value "
                f"{min(u)}"
            )

        max_u = max(u)
        # Store the sets U-U and U+U as arrays of booleans.
        u_minus_u = np.zeros(2 * max_u + 1, dtype=bool)
        u_plus_u = np.zeros(2 * max_u + 1, dtype=bool)
        u_np = np.array(u, dtype=int)

        for i in u_np:
            u_minus_u[i - u_np + max_u] = True
            u_plus_u[i + u_np] = True

        u_minus_u_size = int(np.sum(u_minus_u))
        u_plus_u_size = int(np.sum(u_plus_u))

        if u_minus_u_size > 2 * max_u + 1:
            raise AssertionError(
                "The constraint |U - U| <= 2 max (U) + 1 is not satisfied. Got: "
                f"lhs={u_minus_u_size} but rhs={2 * max_u + 1}."
            )

        return float(np.log(u_minus_u_size / u_plus_u_size) / np.log(2 * max_u + 1) + 1.0)

    # ---------------- BaseEvaluator hooks ----------------

    def parse_output(self, raw_output, problem):
        if raw_output is None:
            raise ValueError("No output.")

        # Keep raw values for integer check in validation.
        u = np.asarray(raw_output).reshape(-1)
        return u

    def validate_output(self, u: np.ndarray, problem) -> None:
        self.assert_all_finite(u, name="U")
        if u.ndim != 1 or u.size == 0:
            raise ValueError(f"Expected a 1D non-empty array, got shape {u.shape}")

        violations = self._count_violations(u)
        if violations > 0:
            raise ValueError(f"{violations} constraint violation(s)")

    def compute_metrics(self, u: np.ndarray, problem):
        # Safe integer cast after validation
        u_int = u.astype(int)
        lb = self.compute_lower_bound(u_int.tolist())
        u_list = u_int.tolist()
        return {
            "lower_bound": float(lb),
            "violations": 0,
            "integer_set": u_list,
        }

    def default_failure_metrics(self, problem):
        return {
            "lower_bound": 0.0,
            "violations": 1,
            "integer_set": [],
        }

    # ---------------- local helper for validation parity ----------------

    def _count_violations(self, u: np.ndarray) -> int:
        """Count constraint violations for U."""
        violations = 0

        # integer-valued check (no rounding allowed)
        if not np.array_equal(u, u.astype(int)):
            violations += 1

        # non-negative
        if np.any(u < 0):
            violations += int(np.sum(u < 0))

        # sorted strictly increasing (also implies uniqueness)
        if not np.all(np.diff(u) > 0):
            violations += 1

        # contains 0
        if not np.any(u == 0):
            violations += 1

        return int(violations)
