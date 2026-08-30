"""Magnus GPU worker for the complete local reference embedding index.

The submitter uploads a text-only tarball of reference/; this worker extracts
it into persistent /data, downloads/loads BGE-M3 with Transformers on GPU, and
invokes the shared research_index.py in HF backend mode. No training or model
release is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

ARCHIVE_SECRET = os.environ.get("INDEX_ARCHIVE_SECRET", "").strip()
OUT = Path(os.environ.get("INDEX_OUT", "/data/magnus/research-index-20260830"))
SOURCE = OUT / "source"
ARCHIVE = Path("/tmp/reference-text.tar.gz")
INDEX_SCRIPT = Path("/tmp/research_index.py")


def marker(stage: str, status: str, **kw: object) -> None:
    print("=== IDX " + stage + " " + status + " " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def safe_extract(path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if target != dest and dest not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(dest)


def main() -> int:
    if not ARCHIVE_SECRET:
        raise SystemExit("INDEX_ARCHIVE_SECRET missing")
    OUT.mkdir(parents=True, exist_ok=True)
    marker("receive", "start")
    subprocess.run(["magnus", "receive", ARCHIVE_SECRET, "-o", str(ARCHIVE)], check=True)
    marker("receive", "done", bytes=ARCHIVE.stat().st_size)
    marker("extract", "start", out=SOURCE)
    SOURCE.mkdir(parents=True, exist_ok=True)
    safe_extract(ARCHIVE, SOURCE)
    marker("extract", "done")
    os.environ["RESEARCH_INDEX_ROOT"] = str(SOURCE)
    os.environ["RESEARCH_EMBED_BACKEND"] = "hf"
    os.environ["RESEARCH_EMBED_MODEL"] = os.environ.get("INDEX_EMBED_MODEL", "BAAI/bge-m3")
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    db = OUT / "index.sqlite3"
    marker("index", "start", db=db, model=os.environ["RESEARCH_EMBED_MODEL"])
    started = time.time()
    proc = subprocess.run([
        "python3", str(INDEX_SCRIPT), "--db", str(db), "index",
        "--embed-limit", "0", "--embed-batch", os.environ.get("INDEX_EMBED_BATCH", "16")
    ], check=False)
    receipt = {"status": "success" if proc.returncode == 0 else "failed",
               "returncode": proc.returncode, "seconds": round(time.time() - started, 1),
               "db": str(db), "source": str(SOURCE),
               "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()}
    (OUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    marker("index", receipt["status"], seconds=receipt["seconds"], db=db)
    print("**IDX_SUMMARY** " + json.dumps(receipt, ensure_ascii=False), flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
