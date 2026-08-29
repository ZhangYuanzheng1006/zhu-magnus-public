#!/usr/bin/env python3
"""Small standard-library disk I/O probe for Magnus workers.

The probe tests each available root independently, records failures by phase, and
always emits a JSON receipt. It deliberately cleans its temporary files in a
finally block.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOTS = ("/data/magnus", "/tmp")
CHUNK = 1024 * 1024
RANDOM_BLOCK = 4096


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def stats(latencies: list[float], byte_count: int = 0) -> dict[str, Any]:
    elapsed = sum(latencies)
    result: dict[str, Any] = {
        "samples": len(latencies),
        "elapsed_s": round(elapsed, 6),
        "p50_ms": round((percentile(latencies, .50) or 0) * 1000, 4) if latencies else None,
        "p95_ms": round((percentile(latencies, .95) or 0) * 1000, 4) if latencies else None,
    }
    if byte_count and elapsed > 0:
        result["MiB_s"] = round(byte_count / (1024 * 1024) / elapsed, 3)
    if elapsed > 0:
        result["IOPS"] = round(len(latencies) / elapsed, 2)
    return result


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def run_root(root: Path, size: int, warmup: int, samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(root), "status": "failed", "phases": {}, "failures": []}
    work: Path | None = None
    try:
        phase = "space_check"
        usage = shutil.disk_usage(root)
        required = size * 2 + RANDOM_BLOCK * max(samples, 1) + 16 * 1024 * 1024
        result["phases"][phase] = {"free_bytes": usage.free, "required_bytes": required}
        if usage.free < required:
            raise RuntimeError(f"free space {usage.free} < required {required}")
        root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="magnus-disk-probe-", dir=str(root)))
        path = work / "payload.bin"
        data = os.urandom(min(CHUNK, size))

        def write_file() -> None:
            remaining = size
            with path.open("wb", buffering=0) as stream:
                while remaining:
                    block = data[: min(len(data), remaining)]
                    stream.write(block)
                    remaining -= len(block)
                os.fsync(stream.fileno())

        def read_file() -> None:
            with path.open("rb", buffering=0) as stream:
                while stream.read(CHUNK):
                    pass

        for phase, operation in (("small_file", lambda: path.write_bytes(b"magnus-disk-probe")),
                                 ("sequential_write", write_file),
                                 ("sequential_read", read_file)):
            try:
                for _ in range(warmup):
                    operation()
                timings = [timed(operation) for _ in range(samples)]
                result["phases"][phase] = stats(timings, size if phase != "small_file" else 0)
            except Exception as exc:
                result["failures"].append({"phase": phase, "error": f"{type(exc).__name__}: {exc}"})

        phase = "random_read_4KiB"
        try:
            if not path.exists() or path.stat().st_size < RANDOM_BLOCK:
                raise RuntimeError("sequential write did not create a 4KiB file")
            offsets = [(i * RANDOM_BLOCK * 7919) % (size - RANDOM_BLOCK + 1)
                       for i in range(max(samples, 1))]
            with path.open("rb", buffering=0) as stream:
                fd = stream.fileno()
                reader = getattr(os, "pread", None)
                def one(offset: int) -> None:
                    if reader:
                        block = reader(fd, RANDOM_BLOCK, offset)
                    else:
                        stream.seek(offset); block = stream.read(RANDOM_BLOCK)
                    if len(block) != RANDOM_BLOCK:
                        raise IOError("short random read")
                for offset in offsets[:warmup]:
                    one(offset)
                timings = [timed(lambda offset=offset: one(offset)) for offset in offsets]
            result["phases"][phase] = stats(timings, RANDOM_BLOCK * len(timings))
        except Exception as exc:
            result["failures"].append({"phase": phase, "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        result["failures"].append({"phase": "space_check", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        result["status"] = "ok" if not result["failures"] else "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--max-mib", type=int, default=256, help="payload size per root (maximum 256 MiB by default)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", default="/data/magnus/closedloop-0828/disk-io-probe-v2")
    args = parser.parse_args()
    if not 1 <= args.max_mib <= 256:
        parser.error("--max-mib must be between 1 and 256")
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be >= 0 and samples must be >= 1")
    receipt: dict[str, Any] = {"probe": "0828_disk_io_probe", "started_unix": time.time(),
                               "payload_mib": args.max_mib, "warmup": args.warmup,
                               "samples": args.samples, "roots": []}
    for raw in args.roots:
        receipt["roots"].append(run_root(Path(raw), args.max_mib * 1024 * 1024,
                                          args.warmup, args.samples))
    receipt["finished_unix"] = time.time()
    receipt["status"] = "ok" if all(x["status"] == "ok" for x in receipt["roots"]) else "failed"
    try:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        receipt["receipt_path"] = str(out / "receipt.json")
    except Exception as exc:
        receipt["output_failure"] = f"{type(exc).__name__}: {exc}"
    print("**SUMMARY_JSON** " + json.dumps(receipt, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0 if receipt["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
