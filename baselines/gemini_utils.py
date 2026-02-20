import asyncio
import contextlib
import logging
import random
import re
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiAlgorithmGenerator:
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config or {}

        api_params = (self.config.get("api_params") or {}).copy()
        sampling = (self.config.get("sampling_params") or {}).copy()

        # Model & retries/concurrency
        self.model_name = (
                api_params.get("gemini_model")
                or self.config.get("gemini_model")
                or "gemini-2.5-flash-lite"
        )
        self.max_retries = int(api_params.get("max_retries", self.config.get("max_retries", 3)))
        self.max_concurrency = int(api_params.get("max_concurrency", self.config.get("max_concurrency", 8)))

        # Sampling / generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=float(sampling.get("temperature", 0.8)),
            top_p=float(sampling.get("top_p", 0.95)),
            max_output_tokens=int(sampling.get("max_tokens", 2048)),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            # 0 disables thinking; -1 for dynamic
            thinking_config=types.ThinkingConfig(thinking_budget=int(sampling.get("thinking_budget", 0))),
        )

        self.last_usage: list[dict[str, int]] = []
        self.generated_completions: list[str] = []
        self.prompts_used: list[str] = []

    async def _call_gemini_async(self, aclient: genai.Client, prompt: str) -> tuple[str, dict]:
        """Async Gemini call with exponential backoff + jitter. Returns (text, usage_dict)."""
        delay = 0.5
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await aclient.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )

                text = getattr(resp, "text", "") or ""
                um = getattr(resp, "usage_metadata", None)
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", 0) if um else 0,
                    "candidates_tokens": getattr(um, "candidates_token_count", 0) if um else 0,
                    "thoughts_tokens": getattr(um, "thoughts_token_count", 0) if um else 0,
                    "total_tokens": getattr(um, "total_token_count", 0) if um else 0,
                    "cached_prompt_tokens": getattr(um, "cached_content_token_count", 0) if um else 0,
                }
                return text, usage

            except Exception as e:
                if attempt == self.max_retries:
                    logger.error("API error (attempt %d/%d): %s", attempt, self.max_retries, e)
                    raise
                sleep_s = min(10.0, delay) + random.uniform(0, min(1.0, delay))
                logger.warning(
                    "API error (attempt %d/%d): %s; retrying in %.2fs",
                    attempt, self.max_retries, e, sleep_s
                )
                await asyncio.sleep(sleep_s)
                delay *= 2

    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        if not response:
            return None

        m = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
        if m:
            return m[0].strip()

        return None

    async def _generate_one(self, aclient: genai.Client, instruction: str):
        text, usage = await self._call_gemini_async(aclient, instruction)
        code = self._extract_code(text)
        return instruction, code, text, usage

    def generate_algorithms(self, problem, n: int) -> list:
        async def _run():
            async with genai.Client(api_key=self.api_key).aio as aclient:
                sem = asyncio.Semaphore(self.max_concurrency)

                async def _task(i: int):
                    # could be stochastic when sequentially sampling, hence why it's in here
                    instruction = problem.generate_instruction()

                    async with sem:
                        try:
                            return await self._generate_one(aclient, instruction)
                        except Exception as e:
                            logger.error("Generation failed for item %d: %s", i, e)
                            return None

                tasks = [asyncio.create_task(_task(i)) for i in range(n)]
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=False)
                finally:
                    # Ensure all tasks are cancelled/awaited if we exit early.
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await t
                return results

        logger.info(
            "Generating %d algorithms with Gemini (model=%s, concurrency=%d)...",
            n, self.model_name, self.max_concurrency
        )
        results = asyncio.run(_run())

        algorithms: list[str] = []
        usage_list: list[dict[str, int]] = []
        text_list: list[str] = []
        prompts_list: list[str] = []
        for res in results:
            if not res:
                continue

            instruction, code, text, usage = res
            if code:
                prompts_list.append(instruction)
                algorithms.append(code)
                usage_list.append(usage or {})
                text_list.append(text)

        self.prompts_used = prompts_list
        self.last_usage = usage_list
        self.generated_completions = text_list
        logger.info("Successfully generated %d/%d algorithms", len(algorithms), n)

        return algorithms
