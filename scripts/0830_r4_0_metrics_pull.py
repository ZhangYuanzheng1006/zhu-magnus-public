"""R4-0: pull every existing metrics artifact from /data back to stdout.

One CPU job cats all known receipts + trainer states with per-file markers so
the local side can rebuild a metrics hub. Read-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path("/data/magnus/closedloop-0828/r4-0-metrics-pull-v1")

FILES = [
    "/data/magnus/models/Qwen3.5-9B-sft-20260828/receipt.json",
    "/data/magnus/models/Qwen3.5-9B-sft-20260828/trainer/checkpoint-150/trainer_state.json",
    "/data/magnus/models/Qwen3.5-9B-sft-20260828-accel/receipt.json",
    "/data/magnus/models/Qwen3.5-9B-sft-20260828-accel/trainer/checkpoint-150/trainer_state.json",
    "/data/magnus/closedloop-0828/r3-3-sft-eval-v4/receipt.json",
    "/data/magnus/closedloop-0828/r3-3-sft-eval-v4/trainer/checkpoint-75/trainer_state.json",
    "/data/magnus/closedloop-0828/r3-3-sft-eval-v4/trainer/checkpoint-150/trainer_state.json",
    "/data/magnus/closedloop-0828/r3-4-toolloop-v2/receipt.json",
    "/data/magnus/closedloop-0828/r3-4-toolloop-v2/success_raw.jsonl",
    "/data/magnus/closedloop-0828/r3-2-teacher-gen-v1/receipt.json",
    "/data/magnus/closedloop-0828/transformers-eval-v0/student-r2-2/receipt.json",
    "/data/magnus/closedloop-0828/transformers-eval-v0/student-probe-v2/receipt.json",
    "/data/magnus/closedloop-0828/disk-io-probe-v2/receipt.json",
    "/data/magnus/closedloop-0828/r3-0-torch27-v4/receipt.json",
    "/data/magnus/closedloop-0828/r3-1-format-autopsy-v1/receipt.json",
    "/data/magnus/closedloop-0828/r4-0-metrics-pull-v1/index.json",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for path in FILES:
        print(f"##### FILE {path} #####", flush=True)
        p = Path(path)
        if not p.exists():
            print("(MISSING)", flush=True)
            index.append({"path": path, "status": "missing"})
            continue
        try:
            text = p.read_text(encoding="utf-8")
            print(text, flush=True)
            index.append({"path": path, "status": "ok", "bytes": p.stat().st_size})
        except Exception as exc:  # noqa: BLE001
            print(f"(READ ERROR {type(exc).__name__}: {exc})", flush=True)
            index.append({"path": path, "status": "error", "error": str(exc)[:200]})
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**PULL_INDEX** " + json.dumps(index, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
