import numpy as np

from random_search_baselines.problems.problem_utils import (
    BaseEvaluator, ProblemConfig, helper
)


class ErdosMinimumOverlapEvaluator(BaseEvaluator):
    """Evaluates Erdős minimum overlap step-function algorithms."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np\nfrom scipy.optimize import minimize"

    # ---------------- helpers (logic kept identical to your string) ----------------

    @staticmethod
    @helper
    def compute_convolution_overlap(step_heights: np.ndarray) -> float:
        """
        Compute the maximum overlap value for a step function.

        Args:
            step_heights: Array of step function heights

        Returns:
            Maximum value of the convolution h * (1 - h)
        """
        # Compute convolution h * (1 - h)
        convolution_values = np.correlate(
            step_heights, 1 - step_heights, mode='full'
        )
        # Return maximum value normalized by len(step_heights) / 2
        return np.max(convolution_values) / len(step_heights) * 2

    @staticmethod
    @helper
    def create_symmetric_sequence(half_seq: np.ndarray) -> np.ndarray:
        """
        Create a symmetric sequence from a half-sequence.

        Args:
            half_seq: Half of the sequence to make symmetric

        Returns:
            Full symmetric sequence: [a1, a2, ..., ak, ..., a2, a1]
        """
        if len(half_seq) == 0:
            return np.array([])

        # Create symmetric sequence
        reversed_seq = half_seq[::-1]
        if len(half_seq) > 1:
            full_seq = np.concatenate((half_seq[:-1], reversed_seq))
        else:
            full_seq = half_seq

        return full_seq

    @staticmethod
    @helper
    def normalize_to_constraint(step_heights: np.ndarray) -> np.ndarray:
        """
        Normalize step heights to satisfy the sum constraint.

        Args:
            step_heights: Array of step heights to normalize

        Returns:
            Normalized step heights where sum equals n_steps / 2.0
        """
        current_sum = np.sum(step_heights)
        target_sum = len(step_heights) / 2.0

        if current_sum == 0:
            return np.ones(len(step_heights)) * (target_sum / len(step_heights))

        return step_heights * (target_sum / current_sum)

    @staticmethod
    @helper
    def validate_step_function(step_heights: np.ndarray) -> bool:
        """
        Validate that step function satisfies all constraints.

        Args:
            step_heights: Array of step heights

        Returns:
            True if valid, False otherwise
        """
        # Check bounds [0, 1]
        if np.any(step_heights < 0) or np.any(step_heights > 1):
            return False

        # Check sum constraint
        expected_sum = len(step_heights) / 2.0
        if not np.isclose(np.sum(step_heights), expected_sum, rtol=1e-9, atol=1e-9):
            return False

        return True

    # ---------------- BaseEvaluator hooks ----------------

    def parse_output(self, raw_output, problem):
        h = np.asarray(raw_output, dtype=float).reshape(-1)
        return h

    def validate_output(self, step_heights: np.ndarray, problem) -> None:
        if step_heights is None:
            raise ValueError("Output is None")

        # Basic sanity + same constraints as validate_step_function
        self.assert_all_finite(step_heights, name="step_heights")
        if step_heights.ndim != 1 or step_heights.size == 0:
            raise ValueError(f"Expected a 1D non-empty array, got shape {step_heights.shape}")

        violations = self._count_violations(step_heights)
        if violations > 0:
            raise ValueError(f"{violations} constraint violation(s)")

    def compute_metrics(self, step_heights: np.ndarray, problem):
        # Same objective as in your code/string
        convolution_values = np.correlate(
            step_heights, 1 - step_heights, mode='full'
        )
        upper_bound = np.max(convolution_values) / len(step_heights) * 2
        return {
            "upper_bound": float(upper_bound),
            "violations": 0,
            "step_heights": step_heights.tolist(),
        }

    def default_failure_metrics(self, problem):
        return {"upper_bound": float("inf"), "violations": 1}

    # ---------------- local helper for validation parity ----------------

    def _count_violations(self, step_heights: np.ndarray) -> int:
        """Count constraint violations for the step function (parity with your original)."""
        violations = 0
        # bounds
        if np.any(step_heights < 0) or np.any(step_heights > 1):
            violations += int(np.sum((step_heights < 0) | (step_heights > 1)))
        # sum == len/2 (single violation flag if not equal, as in your version)
        expected_sum = len(step_heights) / 2.0
        if not np.isclose(np.sum(step_heights), expected_sum, rtol=1e-9, atol=1e-9):
            violations += 1
        return int(violations)
