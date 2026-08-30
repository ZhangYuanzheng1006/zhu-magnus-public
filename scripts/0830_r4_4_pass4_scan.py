"""4.2: pass@4 scan on the R3-3-v4 merged checkpoint (learnable-pool打底).

For each of 200 dev/eval/holdout questions: 4 sampled multi-turn rollouts
(T=0.8, top_p 0.95, 06-protocol loop with real sandbox), pass = >=1 correct
final (CAS/numeric vs gold). Output per-question pass counts -> the P4
learnable-pool construction rule (keep 0<pass@4<1) and the zero-advantage
risk quantification. Sharded: --shard i --nshards N.

Calibration-only on the R3-3 checkpoint (04 §4.2); the formal pool will be
re-measured on the P3 checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VERSION = "r4-4-pass4-scan-v1"
MERGED = os.environ.get("R3_MODEL", "/data/magnus/closedloop-0828/r3-3-sft-eval-v4/merged")
PROBLEMS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
TRAJ = "/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl"
OUT = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r4-4-pass4-v1"))
SHARD = int(os.environ.get("R3_SHARD", "0"))
NSHARDS = int(os.environ.get("R3_NSHARDS", "2"))
N_QUESTIONS = 200
SAMPLES = 4
MAX_ROUNDS = 3
BUDGET = 4096

RECEIPT: dict[str, Any] = {"version": VERSION, "model": MERGED,
                           "shard": SHARD, "nshards": NSHARDS, "samples": SAMPLES}
SPECIAL = re.compile(r"<\|[^<>]*\|>")


def marker(stage: str, status: str = "", **kw: Any) -> None:
    print(f"=== R4-4 {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def flush() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"receipt-shard{SHARD}.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / f"receipt-shard{SHARD}.json")


def sandbox_exec(code: str) -> dict[str, Any]:
    prelude = "import numpy as np\nimport scipy as sp\nimport sympy as sym\n"
    try:
        proc = subprocess.run(["python3", "-c", prelude + code], capture_output=True,
                              text=True, timeout=10.0)
        return {"ok": proc.returncode == 0,
                "output": (proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout))[:8000]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "<timeout>"}


def load_questions() -> list[dict[str, Any]]:
    with Path(PROBLEMS).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in {"dev", "eval", "test", "holdout_family"}:
                system = None
                break
    with Path(TRAJ).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                system = r["messages"][0]["content"]
                break
    qs = []
    with Path(PROBLEMS).open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q.get("split") in {"dev", "eval", "test", "holdout_family"}:
                gold = q["gold"]["canonical_sympy"] if isinstance(q.get("gold"), dict) else str(q.get("gold", ""))
                qs.append({"id": q["id"], "prompt": q["prompt"], "gold": gold})
            if len(qs) >= N_QUESTIONS:
                break
    return system, qs


def align(prompt: str) -> str:
    if prompt.endswith("<think>\n"):
        return prompt + "\n</think>\n\n"
    if prompt.endswith("<|im_start|>assistant\n"):
        return prompt + "<think>\n\n</think>\n\n"
    return prompt


def sym_equiv(text: str, gold: str) -> bool:
    try:
        import sympy as sym
        got = sym.sympify(text.strip())
        want = sym.sympify(gold)
        if sym.simplify(got - want) == 0:
            return True
        f1, f2 = sym.lambdify([], got, "numpy"), sym.lambdify([], want, "numpy")
        return abs(float(f1()) - float(f2())) < 1e-9
    except Exception:  # noqa: BLE001
        return False


def rollout_correct(model: Any, tok: Any, system: str, q: dict[str, Any]) -> tuple[bool, int, float]:
    import torch
    prompt = align(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": q["prompt"]}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False}))
    correct = False
    spent = 0
    for _ in range(MAX_ROUNDS):
        budget = min(1024 if spent == 0 else 2048, BUDGET - spent)
        if budget <= 0:
            break
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        w = ids["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            seq = model.generate(**ids, max_new_tokens=budget, do_sample=True,
                                 temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)[0]
        spent += int(seq[w:].shape[0])
        text = tok.decode(seq[w:], skip_special_tokens=False)
        run_m = re.search(r"<run>\s*(.*?)\s*</run>", text, flags=re.S)
        final_m = re.search(r"<final>\s*(.*?)\s*</final>", text, flags=re.S)
        if final_m:
            if sym_equiv(final_m.group(1), q["gold"]):
                correct = True
            break
        if run_m:
            ex = sandbox_exec(run_m.group(1))
            prompt = (prompt + SPECIAL.split(text)[0] + "<|im_end|>\n<|im_start|>user\n"
                      + f"<output>\n{ex['output']}\n</output><|im_end|>\n"
                      + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        else:
            break
    return correct, spent, 0.0


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    flush()
    marker("env", "done", torch=torch.__version__, shard=f"{SHARD}/{NSHARDS}")

    system, questions = load_questions()
    shard_qs = [q for i, q in enumerate(questions) if i % NSHARDS == SHARD]
    RECEIPT["questions"] = len(shard_qs)
    flush()
    marker("questions", "done", n=len(shard_qs))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False)
    model.eval()
    RECEIPT["load_s"] = round(time.time() - t0, 1)
    flush()
    marker("model_load", "done", seconds=RECEIPT["load_s"])

    per_q = []
    hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    learnable = 0
    for idx, q in enumerate(shard_qs, 1):
        passes = 0
        tok_total = 0
        for s in range(SAMPLES):
            try:
                ok, used, _ = rollout_correct(model, tok, system, q)
                passes += int(ok)
                tok_total += used
            except Exception as exc:  # noqa: BLE001
                marker("rollout", "error", qid=q["id"], sample=s, err=str(exc)[:120])
        hist[passes] = hist.get(passes, 0) + 1
        is_learnable = 0 < passes < SAMPLES
        learnable += int(is_learnable)
        per_q.append({"problem_id": q["id"], "passes": passes, "of": SAMPLES,
                      "learnable": is_learnable, "tokens_total": tok_total})
        RECEIPT["pass_hist"] = hist
        RECEIPT["learnable_count"] = learnable
        RECEIPT["learnable_rate"] = round(learnable / max(1, len(per_q)), 4)
        flush()
        marker("question", "done", i=idx, qid=q["id"], passes=passes)
    (OUT / f"per-question-shard{SHARD}.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in per_q), encoding="utf-8")
    print("**R4-4_SUMMARY** " + json.dumps(
        {"shard": SHARD, "questions": len(per_q), "pass_hist": hist,
         "learnable_rate": RECEIPT["learnable_rate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
