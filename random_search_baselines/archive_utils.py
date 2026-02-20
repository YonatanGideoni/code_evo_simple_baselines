import json
import random

from random_search_baselines.problems.problem_utils import BaseProblem


class ProblemWithArchive:
    def __init__(self, problem: BaseProblem, archive_path: str, k: int = 1, selection_strategy: str = 'random'):
        self._problem = problem

        self.archive = self.load_archive(archive_path)

        self.k = k
        self.selection_strategy = selection_strategy

    @staticmethod
    def load_archive(archive_path: str):
        with open(archive_path, 'rb') as f:
            archive = json.load(f)

        good_answers = [a for a in archive['results'] if a['success']]

        if not good_answers:
            raise Exception(f'No good answers found in archive {archive_path}')

        return good_answers

    def __getattr__(self, name):
        if name == '_problem':
            raise AttributeError()

        return getattr(self._problem, name)

    def select_programs_from_archive(self):
        if self.selection_strategy not in ['random', 'best']:
            raise NotImplementedError(f'Selection strategy {self.selection_strategy} not implemented')

        problem_metric_name = self._problem.config['metric_name']
        problem_conf_metric_name = self._problem.config['conf_metric_name']
        if self.selection_strategy == 'random':
            k = min(self.k, len(self.archive))
            chosen_programs = random.sample(self.archive, k)
        else:  # best
            lower_is_better = self._problem.config.get('lower_is_better', True)
            sorted_archive = sorted(self.archive, key=lambda x: x['metrics'][problem_conf_metric_name],
                                    reverse=not lower_is_better)
            chosen_programs = sorted_archive[:self.k]

        for p in chosen_programs:
            p['metric_name'] = problem_metric_name
            p['metric_value'] = p['metrics'][problem_conf_metric_name]

        return chosen_programs

    def generate_instruction(self):
        base_instruction = self._problem.generate_instruction()

        programs_to_include = self.select_programs_from_archive()

        instr = base_instruction
        instr += '\nHere are some example solutions from previous attempts, please improve on them:'
        for i, program in enumerate(programs_to_include):
            instr += f"\n\n# Solution {i + 1}:\n```\n{program['algorithm_code']}\n```\n"
            instr += f"This solution, solution {i + 1}, got a {program['metric_name']} of {program['metric_value']}.\n"

        instr += '\nPlease now provide your new, better solution.'

        return instr
