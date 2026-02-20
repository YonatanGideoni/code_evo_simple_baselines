import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-algorithms", type=int, default=15, help="Number of algorithms to generate")
    parser.add_argument("--save-results", help="JSON file to save results to", default=None)
    parser.add_argument("--problem", help="Problem name", default=None)

    # model defs
    parser.add_argument("--model", help="Override Gemini model name from config")
    parser.add_argument("--thinking_budget", type=int, help="Override thinking budget (in tokens)")
    parser.add_argument("--max_tokens", type=int, help="Override maximum number of tokens to generate")

    # api calls
    parser.add_argument("--api-config", help="Path to API configuration JSON file (default: api_config.json)")
    parser.add_argument("--api-key", required=True, help="Google AI API key")
    parser.add_argument("--max-retries", type=int, help="Override maximum API retry attempts")

    # execution environment
    parser.add_argument("--eval-workers", type=int, default=8,
                        help="Processes for evaluation (overrides api_config; default 8)")
    parser.add_argument("--gen-workers", type=int, default=8,
                        help="Concurrent requests for generation (overrides api_config; default 8)")
    parser.add_argument('--max_exec_time', type=int, default=None)

    # sequential sampling
    parser.add_argument('--archive_path', type=str, default=None)

    return parser


def get_api_cfg_path(api_config_path) -> str | Path:
    return Path(api_config_path).expanduser() if api_config_path \
        else Path(__file__).resolve().with_name("api_config.json")


def load_api_config(config_path: str | Path) -> dict[str, Any]:
    if config_path is None:
        config_path = get_api_cfg_path(None)

    cfg_path = Path(config_path).expanduser()
    with cfg_path.open("r") as f:
        return json.load(f)


def apply_and_collect_cli_overrides(
        api_params: dict[str, Any],
        args
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Mutates and returns api_params with CLI overrides applied,
    and also returns a dict of the exact CLI overrides used.
    """
    overrides: dict[str, Any] = {}

    mappings: list[tuple[str, str, callable]] = [
        ("max_retries", "max_retries", int),
        ("model", "gemini_model", str),
        ("eval_workers", "num_workers", int),
        ("gen_workers", "max_concurrency", int),
        ("thinking_budget", "thinking_budget", int),
        ("max_tokens", "max_tokens", int),
    ]

    for cli_key, api_key, cast in mappings:
        val = getattr(args, cli_key, None)
        if val is not None:
            casted = cast(val)
            overrides[cli_key] = casted
            api_params[api_key] = casted

    return api_params, overrides


def merge_and_override_configs(
        *,
        prob_config: Any | None,
        api_config: dict[str, Any] | None,
        args,
) -> tuple[dict, dict]:
    """
    Merge (problem config -> api_config -> CLI overrides) and
    build a concise, serializable metadata summary.

    Returns:
      merged_config, meta_summary
    """
    base_config = (getattr(prob_config, "config", None) or {}).copy()

    api_params = dict(base_config.get("api_params", {}))
    api_params.update((api_config or {}).get("api_params", {}) or {})

    sampling_params = (api_config or {}).get("sampling_params", {}) or {}

    api_params, _ = apply_and_collect_cli_overrides(api_params, args)
    sampling_params, cli_overrides = apply_and_collect_cli_overrides(sampling_params, args)

    merged = dict(base_config)
    merged["api_params"] = api_params
    merged["sampling_params"] = sampling_params

    meta = {
        "model_name": api_params.get("gemini_model", "gemini-2.5-flash-lite"),
        "num_algorithms": getattr(args, "num_algorithms", None),
        "api_params": api_params,
        "sampling_params": sampling_params,
        "cli_overrides": cli_overrides,
    }

    return merged, meta


def build_joint_meta(
        *,
        api_config_path: str | Path | None,
        args,
        problems: list[str],
        excluded: list[str],
) -> dict:
    """
    Compute a 'run-level' metadata snapshot
    """
    cfg_path = get_api_cfg_path(api_config_path)

    api_config = load_api_config(cfg_path)
    # Use an empty/dummy problem config here: we want the *common* run settings
    merged_cfg, meta = merge_and_override_configs(
        prob_config=SimpleNamespace(config={}),
        api_config=api_config,
        args=args,
    )
    # meta already has model/api/sampling/overrides
    meta.update({
        "api_config_path": str(cfg_path),
        "problems": problems,
        "skipped": excluded,
        "num_problems": len(problems) if problems is not None else None,
    })
    return meta
