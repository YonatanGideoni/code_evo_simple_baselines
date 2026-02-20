from functools import partial
from typing import Callable
from typing import List, Tuple

from random_search_baselines.problems.long.aime_scaffolding.cache_utils import get_class_hash, save_eval_dataframe_csv
from random_search_baselines.problems.long.aime_scaffolding.eval_utils import query_llm, create_call_limited_query_llm, \
    agent_evaluation

import re
import collections
from typing import Callable, List, Tuple, Dict

import re
from typing import Callable, List, Dict, Optional, Tuple
from collections import Counter

import re
from typing import Callable, List, Dict
from collections import Counter

import re
from collections import Counter
from typing import Callable, List, Tuple

import re
from typing import Callable, List, Tuple
from collections import Counter


class Agent:
    def __init__(
            self,
            query_llm: Callable,
    ):
        """
        Initializes the Self-Refined Ensemble Agent.
        This agent follows a three-phase process:
        1.  GENERATE: Create multiple diverse solution attempts using a few-shot,
            chain-of-thought prompt with a non-zero temperature.
        2.  VERIFY: For each solution, use a separate "verifier" prompt to
            critically analyze the reasoning and assign a confidence score.
        3.  SELECT: Programmatically choose the best answer based on the
            verifier's confidence scores.
        """
        self.query_llm = query_llm
        self.num_solutions_to_generate = 5
        self.output_format_instructions = "On the final line output only the digits of the answer (0-999). Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."

    def forward(self, problem: str) -> tuple[str, float]:
        """
        Executes the Generate-Verify-Select workflow to solve the math problem.

        This method is designed to use a total of 2 * num_solutions_to_generate LLM calls.
        With the default of 5, this uses the maximum of 10 calls.
        """
        total_cost = 0.0
        # 1. --- GENERATION PHASE ---
        # Generate N independent solutions to the problem.
        solutions = []
        for _ in range(self.num_solutions_to_generate):
            system_prompt, task_prompt = self._get_generator_prompt(problem)
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=0.5,  # Use higher temperature for diverse solutions
            )
            solutions.append(response)
            total_cost += cost

        # 2. --- VERIFICATION PHASE ---
        # For each solution, ask a verifier to score its correctness.
        verifications = []
        for solution_text in solutions:
            system_prompt, task_prompt = self._get_verifier_prompt(problem, solution_text)
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=0.0,  # Use zero temperature for deterministic verification
            )
            verifications.append(response)
            total_cost += cost
        # 3. --- SELECTION PHASE ---
        # Programmatically analyze the verifications and select the best answer.
        best_solution = self._select_best_solution(solutions, verifications)

        return best_solution, total_cost

    def _parse_answer(self, text: str) -> str | None:
        """Extracts the 3-digit answer from the \\boxed{...} command."""
        match = re.search(r'\\boxed\{(\d{1,3})\}', text)
        return match.group(1) if match else None

    def _parse_confidence_score(self, text: str) -> int:
        """Extracts the confidence score from the verifier's response."""
        match = re.search(r'Confidence Score: (\d+)/10', text)
        return int(match.group(1)) if match else 0

    def _select_best_solution(self, solutions: List[str], verifications: List[str]) -> str:
        """
        Selects the best solution based on confidence scores.
        - If one solution has a uniquely high score, it is chosen.
        - If there's a tie for the highest score, a majority vote is held
          among the tied solutions.
        - If no clear winner emerges, it falls back to a majority vote over all solutions.
        """
        parsed_answers = [self._parse_answer(s) for s in solutions]
        scores = [self._parse_confidence_score(v) for v in verifications]

        # Filter out solutions where no answer could be parsed
        valid_candidates = [
            (score, answer)
            for score, answer in zip(scores, parsed_answers)
            if answer is not None
        ]
        if not valid_candidates:
            # Fallback: if no solutions have a parsable answer, return a default
            return "\\boxed{000}"
        max_score = max(c[0] for c in valid_candidates)

        # Get all answers that achieved the maximum score
        top_scoring_answers = [answer for score, answer in valid_candidates if score == max_score]
        if len(top_scoring_answers) == 1:
            # A single, clear winner
            winner = top_scoring_answers[0]
        else:
            # Tie for the highest score, so we take a majority vote among the tied candidates
            answer_counts = Counter(top_scoring_answers)
            winner = answer_counts.most_common(1)[0][0]
        return f"\\boxed{{{winner}}}"

    def _get_generator_prompt(self, problem: str) -> Tuple[str, str]:
        """Creates the prompt for the solution generation phase."""
        system_prompt = (
            "You are a world-renowned mathematician, a gold medalist in the "
            "International Mathematical Olympiad. You are a master of creative "
            "problem-solving and rigorous, step-by-step thinking. Your goal is "
            "to solve a challenging math problem from the AIME competition."
        )
        few_shot_example = """
Here is an example of a solution:
[Problem]
Let $S$ be the set of all positive integers $n$ such that $n^2$ is a multiple of both 24 and 108. Which of the following integers are divisors of every integer $n$ in $S$?
(A) 12 (B) 24 (C) 36 (D) 72 (E) 108
[Solution]
Let $n$ be a positive integer in the set $S$.
The condition is that $n^2$ is a multiple of both 24 and 108. This means $n^2$ must be a multiple of the least common multiple (LCM) of 24 and 108.
First, let's find the prime factorization of 24 and 108.
$24 = 8 \times 3 = 2^3 \times 3^1$
$108 = 4 \times 27 = 2^2 \times 3^3$
Now, we find the LCM of 24 and 108. The LCM is found by taking the highest power of each prime factor present in either factorization.
LCM(24, 108) = $2^{\max(3,2)} \times 3^{\max(1,3)} = 2^3 \times 3^3 = 8 \times 27 = 216$.
So, $n^2$ must be a multiple of 216. We can write this as $n^2 = 216k$ for some positive integer $k$.
Let's look at the prime factorization of $n^2$:
$n^2 = (2^3 \times 3^3) \times k$.
For $n^2$ to be a perfect square, the exponent of each prime factor in its prime factorization must be an even number.
Let the prime factorization of $n$ be $n = 2^{a} 3^{b} \ldots$.
Then $n^2 = 2^{2a} 3^{2b} \ldots$. The exponents are all even.
Our current expression for $n^2$ is $2^3 \times 3^3 \times k$. The exponents of 2 and 3 are odd.
To make the exponents even, the integer $k$ must contribute at least one factor of 2 and one factor of 3.
So, $k$ must be of the form $2^1 \times 3^1 \times m^2$ for some integer $m$, to ensure the remaining part is a perfect square.
The smallest possible value for $k$ is $2 \times 3 = 6$.
If $k=6$, then $n^2 = 216 \times 6 = (2^3 \times 3^3) \times (2 \times 3) = 2^4 \times 3^4 = (2^2 \times 3^2)^2 = (4 \times 9)^2 = 36^2$.
So, the smallest possible value for $n$ is 36.
In general, $n^2 = (2^3 \times 3^3) \times (2 \times 3 \times m^2) = 2^4 \times 3^4 \times m^2 = (2^2 \times 3^2 \times m)^2$.
So, $n = 2^2 \times 3^2 \times m = 36m$ for any positive integer $m$.
The set $S$ is the set of all integers of the form $36m$, where $m$ is a positive integer.
$S = \{36, 72, 108, 144, \ldots \}$.
We are looking for the integers that are divisors of *every* integer $n$ in $S$.
This means we are looking for the greatest common divisor (GCD) of all elements in $S$.
The GCD of $\{36, 72, 108, \ldots\}$ is the smallest element, which is 36.
So, any divisor of 36 will be a divisor of every integer in $S$. The question asks which of the given integers are divisors of every $n \in S$. This is equivalent to asking which of the options are divisors of 36.
Let's check the options:
(A) 12 is a divisor of 36.
(B) 24 is not a divisor of 36.
(C) 36 is a divisor of 36.
The problem is phrased as "Which of the following integers are divisors", which is strange. It seems to imply a multiple-choice question format, but AIME is not. Let's assume it wants the largest possible divisor from the list, or the GCD of all elements of S. The GCD of all elements of S is 36.
Let's re-read carefully: "Which of the following integers are divisors of every integer n in S?". AIME answers are integers from 0 to 999. It's likely a mis-formatted problem from another competition. Let's assume the goal is to find the GCD of all elements in S.
The GCD of all elements in S is 36.
The final answer is 36.
\\boxed{036}
"""
        task_prompt = (
            f"Solve the following math problem. Think step-by-step, explaining your "
            f"reasoning and calculations. {self.output_format_instructions}\n\n"
            f"{few_shot_example}\n\n"
            f"[Problem]\n{problem}\n\n[Solution]\n"
        )
        return system_prompt, task_prompt

    def _get_verifier_prompt(self, problem: str, solution: str) -> Tuple[str, str]:
        """Creates the prompt for the solution verification phase."""
        system_prompt = (
            "You are a meticulous and skeptical mathematics professor. Your task is to "
            "critically review a student's proposed solution to an AIME problem. "
            "Do not solve the problem yourself. Your sole purpose is to find any flaws, "
            "logical gaps, or calculation errors in the provided solution. Be harsh in your "
            "assessment if you find even a small error."
        )
        task_prompt = (
            f"Here is the original problem:\n"
            f"'''\n{problem}\n'''\n\n"
            f"Here is the student's proposed solution:\n"
            f"'''\n{solution}\n'''\n\n"
            f"Please perform the following steps:\n"
            f"1. Carefully check each step of the reasoning. Is the logic sound?\n"
            f"2. Verify all calculations. Are there any arithmetical errors?\n"
            f"3. Assess if the student's approach correctly addresses the question asked.\n"
            f"4. Conclude with a confidence score for the solution on a scale of 1 to 10, "
            f"where 1 is 'completely wrong' and 10 is 'highly confident it is correct'. "
            f"Provide your reasoning for the score, then write the score on a new line in the "
            f"exact format: 'Confidence Score: X/10'."
        )
        return system_prompt, task_prompt


def main():
    aime_year = 2025
    split = "full"
    eval_model = "gpt-4.1-nano"
    max_calls = 10
    n_evals_per_q = 10
    n_eval_workers_per_scaffold = 30

    # Create base query_llm function
    base_query_llm = partial(query_llm, model_name=eval_model)

    # Wrap it with call limiting (max 10 calls per forward pass)
    limited_query_llm = create_call_limited_query_llm(
        base_query_llm,
        max_calls=max_calls,
    )

    accuracy, cost_total, n_problems_processed, num_llm_calls, df = agent_evaluation(
        Agent, limited_query_llm, year=aime_year, split=split, n_workers=n_eval_workers_per_scaffold,
        n_evals=n_evals_per_q,
    )

    eval_params = {
        "year": aime_year,
        "split": split,
        "eval_model": eval_model,
        "max_calls": max_calls,
        "n_evals": n_evals_per_q,
        "n_workers": n_eval_workers_per_scaffold,
    }
    agent_hash = get_class_hash(Agent, extra_params=eval_params)

    saved = save_eval_dataframe_csv(df, agent_hash)

    df['problem_rep'] = df.groupby('problem').cumcount()
    print(f"Accuracies: {df.groupby('problem_rep').correct.mean().to_numpy() * 100}")

    print(f'Accuracy: {accuracy}, Total Cost: {cost_total}, '
          f'Num Problems Processed: {n_problems_processed}, Num LLM Calls: {num_llm_calls}')
    print(f'Saved answers to: {saved}')


if __name__ == '__main__':
    main()
