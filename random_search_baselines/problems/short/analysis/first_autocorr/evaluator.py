import numpy as np

from random_search_baselines.problems.problem_utils import (
    BaseEvaluator, EvaluationResult, ProblemConfig, helper
)


class FirstAutocorrEvaluator(BaseEvaluator):
    """Evaluates step-function constructions for the first autocorrelation inequality."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np"

    # ---------------- helpers ----------------

    @staticmethod
    @helper
    def compute_upper_bound(step_heights: np.ndarray) -> float:
        """
        Discrete surrogate upper bound on C1 for equal-width steps on [-1/4, 1/4]:
            C_upper_bound = 2 * len(h) * max(convolve(h, h)) / (sum(h) ** 2)
        """
        convolution = np.convolve(step_heights, step_heights)
        return 2 * len(step_heights) * np.max(convolution) / np.sum(step_heights) ** 2

    # ---------------- BaseEvaluator hooks ----------------

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
        ub = self.compute_upper_bound(step_heights)
        return {
            "upper_bound": float(ub),
            "violations": 0,
            "step_heights": step_heights.tolist(),
        }

    def default_failure_metrics(self, problem):
        return {"upper_bound": float("inf"), "violations": 1}

    # ---------------- local helper ----------------

    def _count_violations(self, h: np.ndarray) -> int:
        v = 0
        if np.any(h < 0):
            v += int(np.sum(h < 0))
        if np.sum(h) <= 0:
            v += 1
        return int(v)
