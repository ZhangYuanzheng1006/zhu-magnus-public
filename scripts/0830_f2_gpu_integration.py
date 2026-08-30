"""F2 GPU integration test (05 review F2; P4 blocking): run the multi-turn
06-protocol rollout executor (f2_multiturn_rollout.py, placed by the submitter)
on the P3 checkpoint-1000 adapter.

This is an integration test, not a capability claim. Acceptance points:
  1. base + LoRA(checkpoint-1000) loads; adapter params are the only trainable;
  2. full loop with REAL sandbox: assistant <run> -> <output> -> assistant,
     env turns masked, token bookkeeping exact (len(mask)==len(ids));
  3. group advantages: mixed-reward group -> nonzero advantage; all-equal
     group -> zero-advantage skip path exercised;
  4. grpo_step: finite loss/kl, update lands on adapter only (ref unchanged);
  5. O-observables emitted per 03b: rewards, zero_advantage_rate, no_final
     rate, rounds, policy_tokens, clip_ratio (=0 by construction, F4).

Reward here is binary plumbing (CAS equivalence on <final>, pass4 logic);
the 03b four-tier grader with claimed-verified detection lands in formal P4
(F1 fix, separate change).
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

VERSION = "f2-gpu-integration-v1"
BASE_MODEL = os.environ.get("P3_MODEL", "/data/magnus/models/Qwen3.5-9B-20260828")
CKPT = os.environ.get("F2_CKPT", "/data/magnus/closedloop-0828/p3-formal-v1/trainer/checkpoint-1000")
OUT = Path(os.environ.get("F2_OUT", "/data/magnus/closedloop-0828/f2-gpu-integration-v1"))
PROBLEMS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
TRAJ = "/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl"
# shard0 pass@4 evidence: 000524 4/4, 000542 1/4 (learnable), 000500/000506 0/4
QIDS_DEFAULT = "vector_div_curl-000524,vector_div_curl-000542,vector_div_curl-000500,vector_div_curl-000506"
QIDS = os.environ.get("F2_QIDS", QIDS_DEFAULT).split(",")
G = 4
BETA = 0.01
LR = 1e-6
MAX_ROUNDS = 3
BUDGET = 4096

RECEIPT: dict[str, Any] = {"version": VERSION, "model": BASE_MODEL, "ckpt": CKPT,
                           "g": G, "beta": BETA, "lr": LR, "qids": QIDS}
SPECIAL = re.compile(r"<\|[^<>]*\|>")

# platform SDK wrapper may clobber PYTHONPATH; script-dir on sys.path is not
# guaranteed either — insert it explicitly (module file is placed alongside
# by the submitter and py_compile-checked before this runs)
sys.path.insert(0, str(Path(__file__).resolve().parent))


def marker(stage: str, status: str = "", **kw: Any) -> None:
    print(f"=== F2I {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def flush_receipt() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "receipt.json")


def sandbox_exec(code: str) -> dict[str, Any]:
    prelude = "import numpy as np\nimport scipy as sp\nimport sympy as sym\n"
    try:
        proc = subprocess.run(["python3", "-c", prelude + code], capture_output=True,
                              text=True, timeout=10.0)
        return {"ok": proc.returncode == 0, "seconds": 0.0,
                "output": (proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout))[:8000]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "seconds": 10.0, "output": "<timeout>"}


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


def make_grader(gold: str):
    def grader(transcript: str) -> tuple[float, dict[str, int]]:
        clean = SPECIAL.sub("", transcript)
        final_m = re.search(r"<final>\s*(.*?)\s*</final>", clean, flags=re.S)
        if not final_m:
            return 0.0, {"no_final": 1}
        if sym_equiv(final_m.group(1), gold):
            return 1.0, {"correct": 1}
        return 0.0, {"wrong": 1}
    return grader


def load_system() -> str:
    with Path(TRAJ).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                return r["messages"][0]["content"]
    raise RuntimeError("no protocol_valid trajectory found for system prompt")


def load_questions() -> dict[str, dict[str, Any]]:
    want = set(QIDS)
    got: dict[str, dict[str, Any]] = {}
    with Path(PROBLEMS).open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q.get("id") in want:
                gold = q["gold"]["canonical_sympy"] if isinstance(q.get("gold"), dict) else str(q.get("gold", ""))
                got[q["id"]] = {"id": q["id"], "prompt": q["prompt"], "gold": gold}
    missing = want - set(got)
    if missing:
        raise RuntimeError(f"requested qids not in problems.jsonl: {sorted(missing)}")
    return got


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from f2_multiturn_rollout import (rollout_one, group_advantages, grpo_step,
                                      align_assistant_prefix)

    t0 = time.time()
    marker("start", VERSION, ckpt=CKPT)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    system = load_system()
    questions = load_questions()
    marker("questions", "loaded", n=len(questions))

    policy = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                  device_map={"": 0})
    policy = PeftModel.from_pretrained(policy, CKPT)
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                               device_map={"": 0}).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vram = torch.cuda.max_memory_allocated() / 2**30
    RECEIPT["load_s"] = round(time.time() - t0, 1)
    RECEIPT["trainable_params"] = n_train
    marker("model_load", "done", seconds=RECEIPT["load_s"], trainable_modules=len(trainable),
           trainable_params=n_train, vram_gb=round(vram, 1))
    if n_train == 0:
        raise RuntimeError("no trainable params after adapter load")

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
    ref_snapshot = next(ref.parameters()).detach().clone()

    groups, steps, episodes = [], [], []
    for qi, qid in enumerate(QIDS):
        q = questions[qid]
        grader = make_grader(q["gold"])
        results = []
        for gi in range(G):
            r = rollout_one(policy, tok, system, q["prompt"], sandbox_exec, grader,
                            max_rounds=MAX_ROUNDS, episode_token_budget=BUDGET)
            checks = {"len_match": len(r.assistant_mask) == len(r.token_ids),
                      "assistant_tokens": sum(r.assistant_mask)}
            if not checks["len_match"] or checks["assistant_tokens"] < 4:
                raise RuntimeError(f"bookkeeping violation {qid} g{gi}: {checks}")
            episodes.append({"qid": qid, "g": gi, "reward": r.reward, "tiers": r.tiers,
                             "rounds": r.rounds, "truncated": r.truncated,
                             "spent": r.meta.get("spent_tokens"), "exec_s": r.exec_seconds})
            marker("episode", "done", qid=qid, g=gi, reward=r.reward, tiers=r.tiers,
                   rounds=r.rounds, truncated=r.truncated, spent=r.meta.get("spent_tokens"))
            results.append(r)
        advs, zero_adv = group_advantages([r.reward for r in results])
        groups.append({"qid": qid, "rewards": [r.reward for r in results],
                       "zero_advantage": zero_adv, "advantages": [round(a, 4) for a in advs]})
        marker("group", "zero_adv" if zero_adv else "mixed", qid=qid,
               rewards=[r.reward for r in results])
        if zero_adv:
            RECEIPT.setdefault("zero_adv_groups", []).append(qid)
            continue
        step = grpo_step(policy, ref, tok, results, advs, beta=BETA, optimizer=opt)
        steps.append({"qid": qid, **{k: (round(v, 6) if isinstance(v, float) else v)
                                     for k, v in step.items()}})
        marker("step", "done", qid=qid, loss=step["loss"], kl=step["kl_mean"],
               policy_tokens=step["policy_tokens"], clip_ratio=step["custom/clip_ratio"])

    ref_unchanged = torch.equal(ref_snapshot, next(ref.parameters()).detach())
    zero_rate = sum(1 for g_ in groups if g_["zero_advantage"]) / max(1, len(groups))
    no_final = sum(1 for e in episodes if "no_final" in e["tiers"]) / max(1, len(episodes))
    correct = sum(1 for e in episodes if "correct" in e["tiers"]) / max(1, len(episodes))
    RECEIPT.update({"episodes": episodes, "groups": groups, "steps": steps,
                    "ref_unchanged": ref_unchanged,
                    "zero_advantage_rate": round(zero_rate, 4),
                    "no_final_rate": round(no_final, 4),
                    "correct_rate": round(correct, 4),
                    "bookkeeping_violations": 0,
                    "wall_s": round(time.time() - t0, 1)})
    flush_receipt()
    marker("summary", "done", groups=len(groups), steps=len(steps),
           zero_adv_rate=RECEIPT["zero_advantage_rate"], no_final_rate=RECEIPT["no_final_rate"],
           correct_rate=RECEIPT["correct_rate"], ref_unchanged=ref_unchanged,
           wall_s=RECEIPT["wall_s"])
    print("**F2I_SUMMARY** " + json.dumps({k: RECEIPT[k] for k in (
        "zero_advantage_rate", "no_final_rate", "correct_rate", "ref_unchanged",
        "bookkeeping_violations", "wall_s")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
