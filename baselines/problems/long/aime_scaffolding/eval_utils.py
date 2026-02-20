import concurrent
import logging
import multiprocessing
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import openai
import pandas as pd
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

_thread_local = threading.local()

M = 1_000_000
_costs_per_token = {
    "gpt-4.1-nano": {"input": 0.1 / M, "output": 0.4 / M},
    "gpt-4.1-mini": {"input": 0.4 / M, "output": 1.6 / M},
    "gpt-4.1": {"input": 2.0 / M, "output": 8.0 / M},
    "gpt-4o-mini": {"input": 0.15 / M, "output": 0.6 / M},
    "o4-mini": {"input": 1.1 / M, "output": 4.4 / M},
    "gemini-2.5-flash-lite": {"input": 0.10 / M, "output": 0.40 / M},
    "gemini-2.5-flash": {"input": 0.30 / M, "output": 2.50 / M},
}

retry_with_jitter = retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=0.5, max=10),
    reraise=True,
)


def _make_usage_object(prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens))


def get_gemini_client():
    if not hasattr(_thread_local, "genai_client"):
        _thread_local.genai_client = genai.Client()  # ensure GOOGLE_API_KEY is configured
    return _thread_local.genai_client


def get_openai_client():
    if not hasattr(_thread_local, "openai_client"):
        _thread_local.openai_client = openai.OpenAI()
    return _thread_local.openai_client


@retry_with_jitter
def query_gemini(model_name, prompt, system, temperature=1.0):
    client = get_gemini_client()

    cfg = genai_types.GenerateContentConfig(temperature=temperature)
    cfg.automatic_function_calling = genai_types.AutomaticFunctionCallingConfig(disable=True)
    if system:
        cfg.system_instruction = system

    resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)

    text = resp.text
    um = resp.usage_metadata

    prompt_tokens = um.prompt_token_count
    candidates_tokens = um.candidates_token_count
    thoughts_tokens = um.thoughts_token_count
    thoughts_tokens = thoughts_tokens if thoughts_tokens is not None else 0

    completion_tokens = candidates_tokens + thoughts_tokens

    return text, _make_usage_object(prompt_tokens, completion_tokens)


@retry_with_jitter
def query_gpt(model_name, prompt, system, temperature=1.):
    if model_name == "gpt-o4-mini":  # doesn't accept non-1 temperature
        temperature = 1.0

    client = get_openai_client()

    if system is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content, response.usage


def query_llm(prompt, system, temperature=0.0, model_name="gpt-4.1-nano"):  # adapted from shinka code
    assert 'gpt' in model_name or 'gemini' in model_name, "Only GPT or Gemini models are supported"

    is_gpt = 'gpt' in model_name
    if is_gpt:
        out_text, resp_usage = query_gpt(model_name, prompt, system, temperature)
    else:
        out_text, resp_usage = query_gemini(model_name, prompt, system, temperature)

    cost = (
            resp_usage.prompt_tokens * _costs_per_token[model_name]["input"]
            + resp_usage.completion_tokens * _costs_per_token[model_name]["output"]
    )
    return out_text, cost


def create_call_limited_query_llm(base_query_llm, max_calls=3):
    """
    Creates a wrapper around query_llm that limits the number of calls
    per forward pass.

    Args:
        base_query_llm: The original query_llm function
        max_calls: Maximum number of calls allowed (default: 3)

    Returns:
        A wrapped query_llm function with call limiting
    """
    thread_local = threading.local()

    def limited_query_llm(*args, **kwargs):
        # Initialize call_count for this thread if it doesn't exist
        if not hasattr(thread_local, "call_count"):
            thread_local.call_count = 0

        if thread_local.call_count >= max_calls:
            class MaxCallsExceededError(Exception):
                """Raised when the maximum number of LLM calls is exceeded."""

                pass

            raise MaxCallsExceededError(
                f"Maximum number of LLM calls ({max_calls}) exceeded"
            )
        thread_local.call_count += 1
        return base_query_llm(*args, **kwargs)

    def reset_calls():
        thread_local.call_count = 0

    def get_call_count():
        return getattr(thread_local, "call_count", 0)

    # Attach reset method to the function
    limited_query_llm.reset_calls = reset_calls
    limited_query_llm.get_call_count = get_call_count

    return limited_query_llm


def agent_evaluation(
        Agent,
        query_llm: Callable,
        year: int = 2024,
        split: str = 'full',
        n_workers: int = 30,
        n_evals: int = 1,
) -> tuple[float, float, int, int, pd.DataFrame]:
    BASE_DIR = Path(__file__).resolve().parent
    math_test_set = pd.read_csv(BASE_DIR / "AIME_Dataset_1983_2025.csv")
    math_test_set = math_test_set[math_test_set["Year"] == year]
    if split != 'full':
        if split == 'odd':
            math_test_set = math_test_set.iloc[::2]
        elif split == 'even':
            math_test_set = math_test_set.iloc[1::2]
        else:
            raise ValueError(f"Invalid split value: {split}. Use 'full', 'odd', or 'even'.")

    math_test_set = pd.concat([math_test_set] * n_evals, ignore_index=True)
    agent = Agent(query_llm)

    results = []
    max_workers = min(n_workers, multiprocessing.cpu_count())
    logger.debug(f"Loaded AIME dataset with {len(math_test_set)} examples")
    logger.debug(f"Running parallel evaluation with {max_workers} workers")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_example, i, example, agent, query_llm): i
            for i, (_, example) in enumerate(math_test_set.iterrows())
        }
        total, correct_count, total_llm_calls, cost_total = 0, 0, 0, 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            total += 1
            try:
                (
                    _idx,
                    problem,
                    response,
                    llm_answer,
                    true_answer,
                    correct,
                    cost,
                    num_llm_calls,
                ) = future.result()
                results.append(
                    {
                        "id": idx,
                        "problem": problem,
                        "response": response,
                        "llm_answer": llm_answer,
                        "true_answer": true_answer,
                        "correct": correct,
                        "cost": cost,
                        "num_llm_calls": num_llm_calls,
                    }
                )
            except Exception as e:
                logger.debug(f"Error processing example {idx}: {e}")
                raise

            cost_total += cost
            if correct:
                correct_count += 1
            total_llm_calls += num_llm_calls
            accuracy = (correct_count / total) * 100
            log_message = (
                f"Step: {total}, LLM answer: {llm_answer}, "
                f"True answer: {true_answer}, "
                f"Accuracy: {accuracy:.2f}%, "
                f"Cost: {cost_total:.4f}, "
                f"LLM calls: {total_llm_calls}, "
                f"Avg LLM calls: {total_llm_calls / total}"
            )
            logger.debug(log_message)

    if total > 0:
        final_accuracy = (correct_count / total) * 100
        logger.debug(f"Complete, final accuracy: {final_accuracy:.2f}%, Cost: {cost_total:.2f}")
        logger.debug(f"Time taken: {time.time() - start_time:.2f} seconds")
        time_per_example = (time.time() - start_time) / total
        logger.debug(f"Time per example: {time_per_example:.2f} seconds")

        df = pd.DataFrame(results)
    else:
        raise ValueError("No examples were processed.")
    return final_accuracy, cost_total, total, total_llm_calls, df


def evaluate_math_correctness(response: str, solution: str) -> tuple[str, str, bool]:
    """Evaluates the correctness of the LLM's response for AIME."""
    llm_answer_str = remove_boxed(last_boxed_only_string(response))
    if llm_answer_str is not None:
        llm_answer_str = llm_answer_str.lstrip("0")
        if llm_answer_str == "":
            llm_answer_str = "0"
    true_answer_str = str(solution)

    true_answer = "" if true_answer_str is None else true_answer_str
    llm_answer = "" if llm_answer_str is None else llm_answer_str

    correct = is_equiv(llm_answer, true_answer)
    return llm_answer, true_answer, correct


def process_example(idx, example, agent, query_llm):
    # Reset call count for each example if using call-limited query_llm
    if hasattr(query_llm, "reset_calls"):
        query_llm.reset_calls()

    problem = example["problem"].strip()
    solution = example["answer"]
    response, cost = agent.forward(problem)
    llm_answer, true_answer, correct = evaluate_math_correctness(response, solution)
    num_llm_calls = query_llm.get_call_count()
    return (
        idx,
        problem,
        response,
        llm_answer,
        true_answer,
        correct,
        cost,
        num_llm_calls,
    )


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        logger.debug("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            logger.debug(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def clean_answer(s):
    # makes no difference but can lead to errors
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("x \\in", "")

    # Remove all \mathbf{...} and replace with just the contents
    s = re.sub(r"\\mathbf\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\textbf\s*{([^}]*)}", r"\1", s)
    return s


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    if not s.startswith(left):
        return None

    assert s[-1] == "}"

    return clean_answer(s[len(left): -1])


def last_boxed_only_string(string: str) -> str:
    """
    Extracts the last LaTeX \\boxed{...} or \\fbox{...} command from a string.
    Handles nested braces. If no \\boxed is found, returns an empty string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
    if idx < 0:
        return ""

    # Find the opening brace
    brace_idx = string.find("{", idx)
    if brace_idx < 0:
        return ""  # No braces, return empty for robustness.

    # Brace matching
    level = 0
    for i in range(brace_idx, len(string)):
        if string[i] == "{":
            level += 1
        elif string[i] == "}":
            level -= 1
            if level == 0:
                return string[idx: i + 1]

    return ""  # Mismatched braces


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing
    # units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    # Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
    if string == "5.5":
        string = "\\frac{11}{2}"
    if "(x - 3)(x + 3)" in string:
        string = string.replace("(x - 3)(x + 3)", "(x+3)(x-3)")

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases
    # fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
