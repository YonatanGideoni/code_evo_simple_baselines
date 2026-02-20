import dis
import hashlib
import io
import json
from pathlib import Path

# save to local answers cache dir, relative to current file
DEF_RESULTS_DIR = Path(__file__).parent / "answers_cache"


def get_class_hash(cls: type, extra_params: dict | None = None) -> str:
    parts: list[str] = []

    parts.append((lambda s: (dis.dis(cls, file=s), s.getvalue())[1])(io.StringIO()))

    if extra_params:
        parts.append(json.dumps(extra_params, sort_keys=True, default=str))

    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            p = p.encode("utf-8")
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()


def save_eval_dataframe_csv(df, agent_hash: str, results_dir: str = DEF_RESULTS_DIR) -> str:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    csv_path = results_path / f"{agent_hash}.csv"

    try:
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    except Exception as e:
        return f"csv save failed: {e}"
