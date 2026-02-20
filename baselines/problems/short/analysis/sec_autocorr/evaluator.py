import numpy as np

from baselines.problems.problem_utils import (
    BaseEvaluator, ProblemConfig, helper
)


class SecondAutocorrEvaluator(BaseEvaluator):
    """Evaluates step-function constructions for the second autocorrelation inequality."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np"

    # ---------------- helpers ----------------

    @staticmethod
    @helper
    def compute_lower_bound(step_heights: np.ndarray) -> float:
        """
        Discrete surrogate lower bound on C2 for equal-width steps on [-1/4, 1/4]:
            Let c = convolve(h, h).
            ||f*f||_2^2  ~=  sum over intervals of (h/3)*(y1^2 + y1*y2 + y2^2)
            ||f*f||_1    ~=  sum |c| / (len(c) + 1)
            ||f*f||_inf   =  max |c|
            Return  ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf)
        """
        convolution = np.convolve(step_heights, step_heights)

        # Calculate the 2-norm squared: ||f*f||_2^2
        num_points = len(convolution)
        x_points = np.linspace(-0.5, 0.5, num_points + 2)
        x_intervals = np.diff(x_points)  # Width of each interval
        y_points = np.concatenate(([0], convolution, [0]))
        l2_norm_squared = 0.0
        for i in range(len(convolution) + 1):  # Iterate through intervals
            y1 = y_points[i]
            y2 = y_points[i + 1]
            h = x_intervals[i]
            # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
            interval_l2_squared = (h / 3) * (y1 ** 2 + y1 * y2 + y2 ** 2)
            l2_norm_squared += interval_l2_squared

        # Calculate the 1-norm: ||f*f||_1
        norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)

        # Calculate the infinity-norm: ||f*f||_inf
        norm_inf = np.max(np.abs(convolution))
        K = l2_norm_squared / (norm_1 * norm_inf)
        return K

    def parse_output(self, raw_output, problem):
        return np.asarray(raw_output, dtype=float).reshape(-1)

    def validate_output(self, step_heights: np.ndarray, problem) -> None:
        if step_heights is None:
            raise ValueError("Output is None")

        self.assert_all_finite(step_heights, name="step_heights")
        if step_heights.ndim != 1 or step_heights.size == 0:
            raise ValueError(f"Expected a 1D non-empty array, got shape {step_heights.shape}")
        violations = self._count_violations(step_heights)
        if violations > 0:
            raise ValueError(f"{violations} constraint violation(s)")

    def compute_metrics(self, step_heights: np.ndarray, problem):
        lb = self.compute_lower_bound(step_heights)
        return {
            "lower_bound": float(lb),
            "violations": 0,
            "step_heights": step_heights.tolist(),
        }

    def default_failure_metrics(self, problem):
        return {"lower_bound": 0.0, "violations": 1}

    # ---------------- local helper ----------------

    def _count_violations(self, h: np.ndarray) -> int:
        v = 0
        if np.any(h < 0):
            v += int(np.sum(h < 0))
        if np.sum(h) <= 0:
            v += 1
        return int(v)
