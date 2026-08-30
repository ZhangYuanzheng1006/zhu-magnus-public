"""Step-0 baseline: run the P3 dev-mini dual-column on the BASE model.

Fills the missing step-0 row the P3 formal run didn't emit (03a §4 R4 修订):
same 50 questions, same system prompt, same multi-turn loop + real sandbox +
greedy decoding as p3_formal_sft.dev_mini_dual, executed on the untrained
base weights. Result is the reference lower bound for readings like
step-500's format 52% / sym 14%.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

VERSION = "p3-step0-baseline-v1"
BASE_MODEL = "/data/magnus/models/Qwen3.5-9B-20260828"
TRAIN_DATA = "/data/magnus/closedloop-0828/p2-20k/sft_trajectories.jsonl"
PROBLEMS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
OUT = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/p3-step0-baseline-v1"))

def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/tmp/p3-step0")
    spec = importlib.util.spec_from_file_location("p3f", "/tmp/p3-step0/0830_p3_formal_sft.py")
    p3f = importlib.util.module_from_spec(spec)
    sys.modules["p3f"] = p3f
    spec.loader.exec_module(p3f)

    receipt: dict = {"version": VERSION, "model": BASE_MODEL, "role": "step-0 baseline"}
    receipt["torch"] = torch.__version__
    receipt["metrics_dir_present"] = bool(os.environ.get("MAGNUS_METRICS_DIR"))
    (OUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    marker = lambda stage, status, **kw: print(
        f"=== P3S0 {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)
    marker("env", "done", torch=torch.__version__, metrics_dir=receipt["metrics_dir_present"])

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False, attn_implementation="sdpa")
    model.eval()
    marker("model_load", "done", seconds=round(time.time() - t0, 1))

    system = None
    with Path(TRAIN_DATA).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                system = r["messages"][0]["content"]
                break
    import hashlib
    import p3f as P
    sys_sha = hashlib.sha256(system.encode()).hexdigest()
    questions = []
    with Path(PROBLEMS).open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q.get("split") in {"dev", "eval", "test", "holdout_family"}:
                gold = q["gold"]["canonical_sympy"] if isinstance(q.get("gold"), dict) else str(q.get("gold", ""))
                questions.append({"id": q["id"], "prompt": q["prompt"], "gold_canonical": gold})
    marker("inputs", "done", sys_sha=sys_sha[:12], questions=len(questions))

    dual = P.dev_mini_dual(model, tok, system, questions)
    receipt["step0_dual"] = dual
    (OUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**P3S0_SUMMARY** " + json.dumps(
        {"format_rate": dual["format_rate"], "sym_rate": dual["sym_rate"], "n": dual["n"]},
        ensure_ascii=False), flush=True)
    marker("receipt", "done", path=str(OUT / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
