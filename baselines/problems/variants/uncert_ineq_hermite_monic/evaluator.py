import numpy as np
from scipy.special import hermite

from baselines.problems.problem_utils import BaseEvaluator, ProblemConfig, helper


class UncertaintyInequalityEvaluator(BaseEvaluator):
    """Evaluates Uncertainty Inequality Hermite coefficient algorithms."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config['problem_parameters']
        self.n_coefficients = self.params['n_coefficients']

    def get_extra_namespace(self, problem):
        return {"n_coefficients": self.n_coefficients}

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np\nfrom scipy.special import hermite"

    @staticmethod
    @helper
    def get_hermite_poly_coeffs(n: int) -> np.ndarray:
        h = hermite(n, monic=True)
        return h.coefficients[::-1]

    @staticmethod
    @helper
    def find_hermite_combination_fast(coeffs: np.ndarray) -> np.ndarray:
        m = len(coeffs)
        degrees = np.arange(0, 4 * m + 4, 4)
        max_degree = degrees[-1]
        poly_coeffs = np.zeros((len(degrees), max_degree + 1))
        for i, deg in enumerate(degrees):
            h_coeffs = UncertaintyInequalityEvaluator.get_hermite_poly_coeffs(deg)
            poly_coeffs[i, :len(h_coeffs)] = h_coeffs
        partial_poly = np.zeros(max_degree + 1)
        for i in range(len(coeffs)):
            partial_poly += coeffs[i] * poly_coeffs[i]
        a = poly_coeffs[-1, 0]
        b = -partial_poly[0]
        last_coeff = 0 if abs(a) < 1e-15 else (b / a)
        final_poly = partial_poly + last_coeff * poly_coeffs[-1]
        if len(final_poly) > 0 and final_poly[-1] < 0:
            final_poly = -final_poly
        return final_poly

    @staticmethod
    @helper
    def get_upper_bound_fast(coeffs: np.ndarray) -> float:
        if np.all(coeffs == 0):
            return 10.0
        try:
            final_poly = UncertaintyInequalityEvaluator.find_hermite_combination_fast(coeffs)
            if len(final_poly) < 3:
                return 10.0
            gq_coeffs = final_poly[2:]
            if len(gq_coeffs) == 0:
                return 10.0
            roots = np.roots(gq_coeffs[::-1])
            real_roots = []
            for root in roots:
                if np.isreal(root):
                    r = float(np.real(root))
                    if r > 1e-10:
                        real_roots.append(r)
            if not real_roots:
                return 10.0
            largest_sign_change = 0.0
            eps = 1e-8
            for r in real_roots:
                left = np.polyval(gq_coeffs[::-1], r - eps)
                right = np.polyval(gq_coeffs[::-1], r + eps)
                if left * right < 0:
                    largest_sign_change = max(largest_sign_change, r)
            if largest_sign_change == 0:
                return 10.0
            return min((largest_sign_change ** 2) / (2 * np.pi), 10.0)
        except Exception:
            return 10.0

    # ---- normal evaluator hooks ----
    def parse_output(self, raw_output, problem):
        coeffs = np.asarray(raw_output, dtype=float).flatten()
        return coeffs

    def validate_output(self, parsed_output, problem) -> None:
        coeffs = parsed_output
        if coeffs.shape != (self.n_coefficients,):
            raise ValueError(f"Shape mismatch: expected ({self.n_coefficients},), got {coeffs.shape}")
        self.assert_all_finite(coeffs, name="coefficients")

    def compute_metrics(self, parsed_output, problem):
        ub = self.get_upper_bound_fast(parsed_output)
        return {"upper_bound": float(ub), "coefficients": [float(c) for c in parsed_output]}

    def default_failure_metrics(self, problem):
        return {"upper_bound": 10.0, "coefficients": [-1, -1, -1]}
