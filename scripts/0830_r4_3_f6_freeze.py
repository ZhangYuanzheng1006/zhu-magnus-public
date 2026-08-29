"""R4-3: F6 input freeze probe (CPU, read-only + snapshot).

1. Replay source availability: stream yahma/alpaca-cleaned via hf-mirror,
   snapshot 1,500 replay rows + 50 held-out probe rows to /data, record
   source + row counts + sha;
2. Inventory /data/magnus/closedloop-0828 for the 20k expansion dataset
   status (P3 launch precondition);
3. Report /data free space.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

OUT = Path("/data/magnus/closedloop-0828/r4-3-f6-freeze-v1")
REPLAY_SRC = "yahma/alpaca-cleaned"
N_REPLAY = 1500
N_PROBE = 50


def sha_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    report: dict = {"version": "r4-3-f6-freeze-v1", "replay_source": REPLAY_SRC}
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. inventory /data for expanded dataset
    base = Path("/data/magnus/closedloop-0828")
    inv = {}
    for p in sorted(base.iterdir()):
        if p.is_dir():
            try:
                n = sum(1 for _ in p.iterdir())
            except Exception:
                n = -1
            inv[p.name] = n
    report["dataset_inventory"] = inv
    for cand in ("p2", "p2-expanded", "data-20k", "p3-data"):
        q = base / cand
        if q.exists():
            jf = list(q.glob("*.jsonl"))
            inv[cand + "::jsonl"] = {f.name: f.stat().st_size for f in jf[:10]}
    # 500-row p2 file line count for scale reference
    p2 = base / "p2" / "sft_trajectories.jsonl"
    if p2.exists():
        report["p2_row_count"] = sum(1 for _ in p2.open(encoding="utf-8"))
    print("**F6 inventory** " + json.dumps(report["dataset_inventory"], ensure_ascii=False), flush=True)

    # 2. replay source via hf-mirror streaming
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from datasets import load_dataset
        ds = load_dataset(REPLAY_SRC, split="train", streaming=True)
        replay_path = OUT / "replay-1500.jsonl"
        probe_path = OUT / "forget-probe-heldout-50.jsonl"
        n_r = n_p = 0
        with replay_path.open("w", encoding="utf-8") as fr, probe_path.open("w", encoding="utf-8") as fp:
            for i, row in enumerate(ds):
                if i < N_PROBE:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_p += 1
                elif i < N_PROBE + N_REPLAY:
                    fr.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_r += 1
                else:
                    break
        report["replay"] = {"status": "ok", "replay_rows": n_r, "probe_rows": n_p,
                            "replay_sha256": sha_of(replay_path), "probe_sha256": sha_of(probe_path)}
    except Exception as exc:  # noqa: BLE001
        report["replay"] = {"status": f"error: {type(exc).__name__}: {exc}"[:300]}
    print("**F6 replay** " + json.dumps(report["replay"], ensure_ascii=False), flush=True)

    # 3. free space
    try:
        import shutil
        u = shutil.disk_usage("/data")
        report["data_free_gb"] = round(u.free / 2**30, 1)
    except Exception:
        pass

    (OUT / "f6-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**F6_SUMMARY** " + json.dumps({k: report[k] for k in ("replay", "data_free_gb", "p2_row_count") if k in report}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
