from typing import Callable, Any

from baselines.problems.problem_utils import ProblemConfig, BaseClassProblem


class AIMEScaffoldingProblem(BaseClassProblem):
    def __init__(self, config: ProblemConfig):
        super().__init__(config)

    def get_class_name(self) -> str:
        return "Agent"

    def get_namespace(self) -> dict[str, Any]:
        return {
            'Callable': Callable
        }
