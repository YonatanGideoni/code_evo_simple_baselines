import inspect
from pathlib import Path
from string import Template
from typing import Dict, Any, List

import numpy as np

from baselines.problems.problem_utils import BaseProblem, ProblemConfig


class ErdosMinimumOverlapProblem(BaseProblem):
    """Defines the Erdős minimum overlap optimization problem."""

    def get_function_name(self) -> str:
        return "find_step_heights"

    def get_namespace(self) -> Dict[str, Any]:
        return {
            'np': np,
            'List': List,
            'max_execution_time': self.params['max_execution_time']
        }
