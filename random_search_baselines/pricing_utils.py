# --- Pricing (per 1M tokens, USD) for Standard (non-batch) requests ---
# Source: https://ai.google.dev/gemini-api/docs/pricing  (text/image/video rows)  [Non-audio]
GEMINI_PRICING = {
    # name key -> dict with 'in', 'out', 'cache' (context caching usage price)
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40, "cache": 0.025},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50, "cache": 0.075},
    # 2.5 Pro has tiers by prompt length (≤200k vs >200k)
    "gemini-2.5-pro": {
        "tiered": True,
        "in_tiers": {"le_200k": 0.625, "gt_200k": 1.25},
        "out_tiers": {"le_200k": 5.00, "gt_200k": 7.50},
        "cache_tiers": {"le_200k": 0.31, "gt_200k": 0.625},
        "threshold_prompt_tokens": 200_000
    },
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40, "cache": 0.025},
}


def _normalize_model_key(model_name: str) -> str:
    """Map preview/suffixed names back to the stable key."""
    m = model_name.lower()
    # Order matters: check more specific names first
    if m.startswith("gemini-2.5-flash-lite"):
        return "gemini-2.5-flash-lite"
    if m.startswith("gemini-2.5-flash"):
        return "gemini-2.5-flash"
    if m.startswith("gemini-2.5-pro"):
        return "gemini-2.5-pro"
    if m.startswith("gemini-2.0-flash"):
        return "gemini-2.0-flash"
    return m  # fallback (may be unknown)


def _get_unit_prices(model_name: str, prompt_tokens: int) -> tuple[float, float, float] | None:
    """
    Return (input_price_per_1M, output_price_per_1M, cache_price_per_1M) for this call.
    Applies 2.5-Pro tiering by *per-request* prompt tokens. Returns None if unknown.
    """
    key = _normalize_model_key(model_name)
    p = GEMINI_PRICING.get(key)
    if not p:
        return None
    if p.get("tiered"):
        thr = p["threshold_prompt_tokens"]
        tier = "le_200k" if prompt_tokens <= thr else "gt_200k"
        return (p["in_tiers"][tier], p["out_tiers"][tier], p["cache_tiers"][tier])
    return (p["in"], p["out"], p["cache"])


def _cost_from_usage(model_name: str, usage_counts: dict) -> dict:
    """
    Compute USD costs for a single request given a usage dict with:
      prompt_tokens, output_tokens, cached_prompt_tokens
    Uses Standard (non-batch) pricing & *text/image/video* token rows.
    Note: excludes any context-cache *storage* fees per hour (needs time data).
    """
    pt = int(usage_counts.get("prompt_tokens", 0))
    out = int(usage_counts.get("output_tokens", 0))
    cached = int(usage_counts.get("cached_prompt_tokens", 0))

    prices = _get_unit_prices(model_name, pt)
    if prices is None:
        return {"input_usd": 0.0, "output_usd": 0.0, "cache_usd": 0.0, "total_usd": 0.0}

    in_per_m, out_per_m, cache_per_m = prices
    billable_input = max(pt - cached, 0)

    input_usd = (billable_input / 1_000_000.0) * in_per_m
    cache_usd = (cached / 1_000_000.0) * cache_per_m
    output_usd = (out / 1_000_000.0) * out_per_m
    total_usd = input_usd + cache_usd + output_usd
    return {
        "input_usd": input_usd,
        "output_usd": output_usd,
        "cache_usd": cache_usd,
        "total_usd": total_usd
    }
