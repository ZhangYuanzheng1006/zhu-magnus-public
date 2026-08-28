#!/usr/bin/env python3
"""P1 v0 driver: generate splits, verify gold, run sandbox regression, emit metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from generators import generate_dataset, write_jsonl
from verifier import run_regression, verify_generated_problem
from metrics import emit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/magnus/closedloop-0828/p1")
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--dev", type=int, default=100)
    p.add_argument("--holdout", type=int, default=50)
    p.add_argument("--secret", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260828)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = generate_dataset(train=args.train, dev=args.dev, holdout=args.holdout, secret=args.secret, seed=args.seed)
    write_jsonl(rows, str(out / "problems.jsonl"))
    split_counts = Counter(row["split"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)
    print("generated:", dict(split_counts))
    print("families:", dict(family_counts))

    verify_rows = [verify_generated_problem(row) for row in rows]
    with (out / "gold_verification.jsonl").open("w", encoding="utf-8") as f:
        for result in verify_rows:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    consistent = sum(bool(r["consistent"]) for r in verify_rows)
    uncertain = sum(bool(r["uncertain"]) for r in verify_rows)
    consistency_rate = consistent / max(1, len(verify_rows))
    print(f"gold consistency: {consistent}/{len(verify_rows)} = {consistency_rate:.4f}")
    print(f"gold uncertain: {uncertain}/{len(verify_rows)} = {uncertain/max(1,len(verify_rows)):.4f}")
    emit("gen.problems", len(rows), kind="counter", step=0, step_domain="generation", unit="problems", labels={"phase": "p1"})
    emit("verify.pass_rate", consistency_rate, unit="percent", step=0, step_domain="generation", labels={"phase": "p1"})
    emit("verify.uncertain_rate", uncertain / max(1, len(verify_rows)), unit="percent", step=0, step_domain="generation", labels={"phase": "p1"})

    regression = run_regression()
    (out / "sandbox_regression.json").write_text(json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sandbox regression:", regression["passed"], "/", regression["total"], "all_pass=", regression["all_pass"])
    report = {
        "generator_version": "p1-v0.1",
        "seed": args.seed,
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "gold_total": len(verify_rows),
        "gold_consistent": consistent,
        "gold_consistency_rate": consistency_rate,
        "gold_uncertain": uncertain,
        "sandbox_regression": regression,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (out / "p1_receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("receipt:", out / "p1_receipt.json")
    return 0 if regression["all_pass"] and consistency_rate >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
