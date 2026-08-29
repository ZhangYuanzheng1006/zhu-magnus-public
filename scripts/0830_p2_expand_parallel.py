"""R4-4: parallel P2 data expansion (19,800 trajectories, multiprocessing).

Sequential run_p2 costs ~4-6 s/row (fresh python3 process + sympy import per
sandbox call) => 20h+ for 19,800 rows. This wrapper parallelizes
make_trajectory over a process pool (each worker still runs the real sandbox
subprocess; order preserved via Pool.map on the seeded dataset).

Output: /data/magnus/closedloop-0828/p2-20k/sft_trajectories.jsonl (+ receipt),
same schema as P2. protocol-valid rows only are the P3 filter's job downstream;
we keep all rows and report yield.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

OUT = Path("/data/magnus/closedloop-0828/p2-20k")
COUNT = 19800
SEED = 20260830
WORKERS = int(os.environ.get("P2X_WORKERS", "24"))

_marker_ok = True


def marker(stage: str, status: str, **kw) -> None:
    print(f"=== P2X {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def _make(row):
    from run_p2 import make_trajectory
    return make_trajectory(row)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/tmp/p2-runtime")
    from run_p2 import generate_dataset, SYSTEM_PROMPT
    import hashlib

    marker("gen_problems", "start", count=COUNT, seed=SEED)
    rows = generate_dataset(train=COUNT, dev=0, holdout=0, secret=0, seed=SEED)
    marker("gen_problems", "done", rows=len(rows))

    t0 = time.time()
    marker("manufacture", "start", workers=WORKERS)
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        trajectories = []
        for i, traj in enumerate(pool.imap(_make, rows, chunksize=16)):
            trajectories.append(traj)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.time() - t0)
                marker("manufacture", "progress", done=i + 1, rate=round(rate, 2))
    wall = time.time() - t0
    marker("manufacture", "done", rows=len(trajectories), wall_s=round(wall, 1))

    out_path = OUT / "sft_trajectories.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    legal = sum(t["protocol_valid"] for t in trajectories)
    executed = sum(t["run"]["execution"]["ok"] for t in trajectories)
    receipt = {"count": len(trajectories), "protocol_valid": legal,
               "protocol_valid_rate": legal / max(1, len(trajectories)),
               "executed": executed,
               "synthetic_yield": executed / max(1, len(trajectories)),
               "wall_s": round(wall, 1), "workers": WORKERS,
               "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
    (OUT / "p2_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**P2X_SUMMARY** " + json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0 if legal == len(trajectories) and executed == len(trajectories) else 2


if __name__ == "__main__":
    raise SystemExit(main())
