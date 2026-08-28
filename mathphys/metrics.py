"""Small fail-open emitter for Magnus Metrics Protocol v1."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def emit(name: str, value: float, *, kind: str = "gauge", step: int | None = None,
         step_domain: str | None = None, unit: str | None = None,
         labels: dict[str, str] | None = None, filename: str = "rank-0.jsonl") -> bool:
    """Append one legal metric point; missing metrics dir never breaks the task."""
    try:
        value = float(value)
        if not math.isfinite(value) or kind not in {"gauge", "counter"}:
            return False
        directory = os.environ.get("MAGNUS_METRICS_DIR")
        if not directory or not os.path.isdir(directory):
            return False
        point: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "value": value,
            "time_unix_ms": int(time.time() * 1000),
        }
        if step is not None:
            point["step"] = int(step)
            point["step_domain"] = step_domain or "global"
        if unit is not None:
            point["unit"] = unit
        if labels:
            point["labels"] = {str(k): str(v) for k, v in labels.items()}
        with (Path(directory) / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(point, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
        return True
    except Exception:
        return False
