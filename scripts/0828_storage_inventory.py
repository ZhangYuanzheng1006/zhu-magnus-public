#!/usr/bin/env python3
"""Read-only filesystem inventory for Magnus CPU storage.

The scanner never opens files, never follows symbolic links, and records a
bounded, time-limited inventory. Individual lstat/scandir failures are kept in
its receipt so a partially inaccessible tree can still be reported.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOTS = ("/data", "/data/magnus")
SENSITIVE_WORDS = ("secret", "token", "password", "passwd", "credential", "private", "apikey", "api_key", ".ssh", "id_rsa", "key")


def sensitive(name: str) -> bool:
    low = name.lower()
    return any(word in low for word in SENSITIVE_WORDS)


def error_record(errors: list[dict[str, Any]], operation: str, path: str, exc: BaseException) -> None:
    errors.append({"operation": operation, "path": path, "error": f"{type(exc).__name__}: {exc}"})


def df_snapshot(path: Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        result = {"path": str(path), "total_bytes": usage.total, "used_bytes": usage.used,
                  "free_bytes": usage.free, "used_ratio": round(usage.used / usage.total, 6) if usage.total else None}
    except Exception as exc:
        error_record(errors, "df", str(path), exc)
        result = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    # df -P provides filesystem/mount identity where available, without reading file contents.
    try:
        proc = subprocess.run(["df", "-P", "-k", "--", str(path)], capture_output=True, text=True,
                              timeout=3, check=False)
        lines = proc.stdout.strip().splitlines()
        if len(lines) >= 2:
            fields = lines[-1].split()
            if len(fields) >= 6:
                result["filesystem"] = fields[0]
                result["mount"] = " ".join(fields[5:])
    except Exception as exc:
        error_record(errors, "df_command", str(path), exc)
    return result


def scan_root(root: Path, deadline: float, max_entries: int) -> dict[str, Any]:
    started = time.monotonic()
    errors: list[dict[str, Any]] = []
    types: Counter[str] = Counter()
    first_level: dict[str, dict[str, int]] = {}
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    count = 0
    truncated = False

    def visit(path: Path, level_one: str | None = None) -> None:
        nonlocal count, truncated
        if count >= max_entries or time.monotonic() >= deadline:
            truncated = True
            return
        try:
            info = path.lstat()
        except Exception as exc:
            error_record(errors, "lstat", str(path), exc)
            return
        count += 1
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            types["symlink"] += 1
            return
        if stat.S_ISDIR(mode):
            types["directory"] += 1
            record = {"path": str(path), "size_bytes": 0, "entries": 0}
            dirs.append(record)
            try:
                with os.scandir(path) as stream:
                    for entry in stream:
                        if count >= max_entries or time.monotonic() >= deadline:
                            truncated = True
                            break
                        before = count
                        visit(Path(entry.path), level_one if level_one is not None else entry.name)
                        if count > before and dirs:
                            record["entries"] += 1
            except Exception as exc:
                error_record(errors, "scandir", str(path), exc)
            return
        if stat.S_ISREG(mode):
            types["regular"] += 1
            suffix = path.suffix.lower() or "[no_extension]"
            types[f"file_type:{suffix}"] += 1
            item = {"path": str(path), "name": path.name, "size_bytes": info.st_size,
                    "mode": stat.filemode(mode), "uid": info.st_uid, "gid": info.st_gid,
                    "mtime_ns": info.st_mtime_ns, "sensitive_name": sensitive(path.name)}
            files.append(item)
            if level_one is not None:
                bucket = first_level.setdefault(level_one, {"entries": 0, "files": 0, "bytes": 0})
                bucket["entries"] += 1; bucket["files"] += 1; bucket["bytes"] += info.st_size
            return
        types["other"] += 1

    root_df = df_snapshot(root, errors)
    visit(root)
    # Directory totals are computed from direct file records, then accumulated bottom-up.
    by_path = {d["path"]: d for d in dirs}
    for item in files:
        parent = Path(item["path"]).parent
        while str(parent).startswith(str(root)) and str(parent) in by_path:
            by_path[str(parent)]["size_bytes"] += item["size_bytes"]
            if parent == root: break
            parent = parent.parent
    return {"root": str(root), "status": "ok" if not errors else "partial", "df": root_df,
            "entries_seen": count, "truncated": truncated, "elapsed_s": round(time.monotonic() - started, 4),
            "first_level": first_level, "directories": sorted(dirs, key=lambda x: x["size_bytes"], reverse=True),
            "top_files": sorted(files, key=lambda x: x["size_bytes"], reverse=True)[:30],
            "files": files, "file_types": dict(sorted(types.items())), "errors": errors}


def safe_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    roots = []
    for item in receipt["roots"]:
        roots.append({"root": item["root"], "status": item["status"], "df": item["df"],
                      "entries_seen": item["entries_seen"], "truncated": item["truncated"],
                      "directory_top": [{"path_category": "sensitive" if sensitive(Path(x["path"]).name) else "directory",
                                         "size_bytes": x["size_bytes"]} for x in item["directories"][:10]],
                      "largest_files": [{"path_category": "sensitive" if x.get("sensitive_name") else "file",
                                         "size_bytes": x["size_bytes"]} for x in item["top_files"]],
                      "file_types": item["file_types"],
                      "error_categories": dict(Counter(e["operation"] for e in item["errors"])),
                      "sensitive_name_count": sum(1 for f in item["files"] if f.get("sensitive_name"))})
    return {"probe": receipt["probe"], "status": receipt["status"], "roots": roots,
            "total_errors": sum(len(x["errors"]) for x in receipt["roots"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--max-entries", type=int, default=100000)
    parser.add_argument("--time-limit-s", type=float, default=120.0)
    parser.add_argument("--out", default="/data/magnus/closedloop-0828/storage-inventory-0828")
    args = parser.parse_args()
    if args.max_entries < 1 or args.time_limit_s <= 0: parser.error("limits must be positive")
    receipt: dict[str, Any] = {"probe": "0828_storage_inventory", "started_unix": time.time(),
                               "limits": {"max_entries": args.max_entries, "time_limit_s": args.time_limit_s}, "roots": []}
    deadline = time.monotonic() + args.time_limit_s
    for raw in args.roots:
        if time.monotonic() >= deadline: break
        receipt["roots"].append(scan_root(Path(raw), deadline, args.max_entries))
    receipt["finished_unix"] = time.time()
    receipt["status"] = "ok" if receipt["roots"] and all(x["status"] == "ok" for x in receipt["roots"]) else "partial"
    try:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        receipt["receipt_path"] = str(out / "receipt.json")
        (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        receipt["output_failure"] = f"{type(exc).__name__}: {exc}"
    print("**SUMMARY_JSON** " + json.dumps(safe_summary(receipt), ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
