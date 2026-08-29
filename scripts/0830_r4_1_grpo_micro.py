"""R4-1: GRPO micro-run (6 steps, G=4) on the R3-3 merged checkpoint.

Purpose (08 §4.2): produce the FIRST REAL VALUES for every GRPO observable in
the design book (reward tiers, group_valid_size, zero_advantage_rate, KL,
entropy, completion length, truncation, clip_ratio, timings) - not merely
"pipeline runs".

Honest scope: TRL GRPOTrainer is single-turn; the production multi-turn
run/output loop is a P4 plugin. The reward function here still executes the
generated <run> code in the real sandbox and grades <final> against gold on
the R3 tier scale, so rewards are real, not mocked.

Config (07 §3 defaults): G=4, lr=1e-6, beta=0.01, T=0.8, top_p=0.95,
clip 0.2, updates=1, completion 512 (micro budget; production 2048).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

VERSION = "r4-1-grpo-micro-v5"
MERGED = os.environ.get("R3_MODEL", "/data/magnus/closedloop-0828/r3-3-sft-eval-v4/merged")
PROBLEMS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
OUT = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r4-1-grpo-micro-v1"))
SYSTEM_SHA = "8ed1122a47ae089b1f577d61ad906cf4f7aa5f39627bfef7b6bf2afe79be3217"
G = 4
STEPS = 6
PROMPTS_PER_STEP = 4
MAX_COMPLETION = 512
LR = 1e-6
BETA = 0.01
TEMP = 0.8
TOP_P = 0.95

RECEIPT: dict[str, Any] = {"version": VERSION, "model": MERGED, "config": {
    "G": G, "steps": STEPS, "prompts_per_step": PROMPTS_PER_STEP,
    "max_completion": MAX_COMPLETION, "lr": LR, "beta": BETA,
    "temperature": TEMP, "top_p": TOP_P, "updates_per_rollout": 1}}
TRACE: list[dict[str, Any]] = []
SPECIAL = re.compile(r"<\|[^<>]*\|>")


def marker(stage: str, status: str = "", **kw: Any) -> None:
    line = f"=== R4-1 {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items())
    print(line + " ===", flush=True)


def flush() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "receipt.json")


def sandbox_exec(code: str, timeout_s: float = 10.0) -> dict[str, Any]:
    prelude = "import numpy as np\nimport scipy as sp\nimport sympy as sym\n"
    try:
        proc = subprocess.run(["python3", "-c", prelude + code], capture_output=True,
                              text=True, timeout=timeout_s)
        out = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
        return {"ok": proc.returncode == 0, "output": out[:8000], "rc": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "<timeout>", "rc": 124}


def load_questions() -> list[dict[str, Any]]:
    rows = []
    with Path(PROBLEMS).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") in {"dev", "eval", "test", "holdout_family"}:
                rows.append(row)
    if len(rows) < PROMPTS_PER_STEP * STEPS:
        raise RuntimeError(f"need {PROMPTS_PER_STEP*STEPS} questions, got {len(rows)}")
    return rows[:PROMPTS_PER_STEP * STEPS]


def render_prompt(tok: Any, system: str, user: str) -> str:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False})
    if prompt.endswith("<think>\n"):
        prompt += "\n</think>\n\n"
    elif prompt.endswith("<|im_start|>assistant\n"):
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def grade(final_text: str, gold_canonical: str) -> tuple[str, float, str]:
    """R3 tier scale. Returns (tier, reward, note)."""
    lower = final_text.strip().lower()
    if not final_text.strip():
        return ("format_fail", 0.0, "empty final")
    if "无法确定" in final_text or "无法判断" in final_text:
        return ("abstain", 0.05, "declared-uncertain (solvable question)")
    try:
        import sympy as sym
        got = sym.sympify(final_text.strip())
        want = sym.sympify(gold_canonical)
        diff = sym.simplify(got - want)
        if diff == 0:
            return ("correct", 1.0, "CAS equal")
        num_ok = False
        try:
            f1, f2 = sym.lambdify([], got, "numpy"), sym.lambdify([], want, "numpy")
            num_ok = abs(float(f1()) - float(f2())) < 1e-9
        except Exception:  # noqa: BLE001
            pass
        if num_ok:
            return ("correct", 1.0, "numeric equal")
        claimed = any(k in lower for k in ("已验证", "验证", "恒等", "成立"))
        if claimed:
            return ("wrong_claimed_verified", -0.2, "wrong but claims verified")
        return ("wrong", 0.0, "CAS mismatch")
    except Exception as exc:  # noqa: BLE001
        claimed = any(k in lower for k in ("已验证", "验证", "恒等", "成立"))
        if claimed:
            return ("wrong_claimed_verified", -0.2, f"unparseable but claims verified: {exc}")
        return ("wrong", 0.0, f"unparseable: {type(exc).__name__}")


def make_reward_fn(system: str, tok: Any):
    def reward_corr(completions, gold_canonical=None, **kwargs):
        out = []
        for comp, gold in zip(completions, gold_canonical):
            t0 = time.time()
            comp_clean = SPECIAL.sub("", comp)
            run_m = re.search(r"<run>\s*(.*?)\s*</run>", comp_clean, flags=re.S)
            final_m = re.search(r"<final>\s*(.*?)\s*</final>", comp_clean, flags=re.S)
            exec_s = None
            exec_out = None
            if run_m:
                ex = sandbox_exec(run_m.group(1))
                exec_s = round(time.time() - t0, 3)
                exec_out = ex["output"][:120]
            if final_m:
                tier, reward, note = grade(final_m.group(1), gold)
            elif run_m:
                tier, reward, note = ("no_final", 0.0, "run without final")
            else:
                tier, reward, note = ("format_fail", 0.0, "no run, no final")
            truncated = not comp_clean.rstrip().endswith("<|im_end|>") and len(comp_clean) >= MAX_COMPLETION - 4
            TRACE.append({"tier": tier, "reward": reward, "note": note,
                          "completion_chars": len(comp_clean), "truncated_guess": bool(truncated),
                          "exec_seconds": exec_s, "exec_head": exec_out,
                          "final_head": final_m.group(1)[:80] if final_m else None})
            out.append(float(reward))
        return out
    reward_corr.__name__ = "reward_corr"
    return reward_corr



def _run_stream(cmd, timeout, env=None):
    import subprocess, threading, time
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    def pump(src, tag):
        for line in src:
            print(f"{tag}| {line.strip()[:150]}", flush=True)
    threading.Thread(target=pump, args=(proc.stdout, "pip"), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, "piperr"), daemon=True).start()
    stop = threading.Event()
    def beat():
        while not stop.wait(45):
            print(f"[heartbeat] alive_s={int(time.time()-t0)}", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    rc = proc.wait(timeout=timeout)
    stop.set()
    return rc


def ensure_torch27_venv() -> str | None:
    """TRL GRPOTrainer needs FSDPModule (torch>=2.6); image torch is 2.5.1.
    Bootstrap a /dev/shm venv with torch 2.7.1 cu126 (validated by R3-0 v3/v4:
    download ~494s via mirror, local install ~72s) and re-exec under it."""
    try:
        from torch.distributed.fsdp import FSDPModule  # noqa: F401
        return None
    except Exception:
        pass
    import shutil
    import subprocess
    venv = Path("/dev/shm/r4-1/venv")
    py = venv / "bin" / "python"
    marker("bootstrap", "start", reason="FSDPModule missing in torch 2.5.1; v3 adds torchvision 0.22.1 ABI match")
    try:
        free = shutil.disk_usage("/dev/shm").free
    except Exception:
        free = 0
    if free < 12 * 2**30:
        venv = Path("/tmp/r4-1/venv")
        py = venv / "bin" / "python"
    env = dict(os.environ)
    env["PIP_NO_COMPILE"] = "1"
    if not py.exists():
        subprocess.run(["python3", "-m", "venv", "--system-site-packages", str(venv)],
                       check=True, timeout=300)
        wheels = Path("/tmp/r4-1/wheels")
        wheels.mkdir(parents=True, exist_ok=True)
        rc = _run_stream([str(venv / "bin" / "pip"), "download", "torch==2.7.1",
                          "torchvision==0.22.1", "torchaudio==2.7.1",
                          "-d", str(wheels), "--timeout", "60", "--retries", "3",
                          "--progress-bar", "off"], 2400, env)
        if rc != 0:
            raise RuntimeError("torch wheel download failed")
        rc = _run_stream([str(venv / "bin" / "pip"), "install", "--no-index",
                          "--find-links", str(wheels), "torch==2.7.1",
                          "torchvision==0.22.1", "torchaudio==2.7.1", "--progress-bar", "off"], 1200, env)
        if rc != 0:
            raise RuntimeError("torch local install failed")
    marker("bootstrap", "done", venv=str(venv))
    return str(py)


def main() -> int:
    py27 = ensure_torch27_venv()
    if py27:
        marker("reexec", "start", py=py27)
        os.execv(py27, [py27, __file__])
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    flush()
    marker("env", "done", torch=torch.__version__)

    questions = load_questions()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.05)
    model = get_peft_model(model, lora)
    RECEIPT["model_load_s"] = round(time.time() - t0, 1)
    RECEIPT["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flush()
    marker("model_load", "done", seconds=RECEIPT["model_load_s"],
           trainable=RECEIPT["trainable"])

    # system prompt: the P2 training system (sha 8ed1122a), re-read from the
    # trajectory file — problems.jsonl does not embed it.
    system = None
    with Path("/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("protocol_valid") is True:
                system = row["messages"][0]["content"]
                break
    RECEIPT["system_found"] = system is not None
    flush()
    marker("system_prompt", "done", found=system is not None)

    rows = []
    for q in questions:
        rows.append({
            "prompt": render_prompt(tok, system, q["prompt"]),
            "gold_canonical": str(q["gold"]["canonical_sympy"]) if isinstance(q.get("gold"), dict) else str(q.get("gold", "")),
            "problem_id": q["id"],
        })
    ds = Dataset.from_list(rows)
    RECEIPT["questions"] = len(ds)
    flush()
    marker("dataset", "done", rows=len(ds))

    import dataclasses
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    kw: dict[str, Any] = dict(
        output_dir=str(OUT / "trainer"),
        per_device_train_batch_size=PROMPTS_PER_STEP * G,
        num_generations=G,
        max_completion_length=MAX_COMPLETION,
        learning_rate=LR,
        beta=BETA,
        temperature=TEMP,
        top_p=TOP_P,
        max_steps=STEPS,
        num_iterations=1,
        bf16=True,
        logging_steps=1,
        report_to=[],
        seed=20260830,
    )
    if "gradient_checkpointing" in fields:
        kw["gradient_checkpointing"] = True
    if "use_vllm" in fields:
        kw["use_vllm"] = False
    dropped = sorted({"use_vllm", "gradient_checkpointing"} - fields)
    RECEIPT["grpo_config_fields_used"] = sorted(kw.keys())
    RECEIPT["grpo_config_fields_dropped"] = dropped
    flush()
    marker("grpo_config", "done", dropped=",".join(dropped) if dropped else "none")
    cfg = GRPOConfig(**kw)

    reward_fn = make_reward_fn(system, tok)
    trainer = GRPOTrainer(model=model, args=cfg, reward_funcs=[reward_fn],
                          train_dataset=ds, processing_class=tok)

    from transformers.trainer_callback import TrainerCallback

    class StepDump(TrainerCallback):
        def on_step_end(self, args, state, control, **kw2):
            keys = ["loss", "kl", "entropy", "clip_ratio", "completions/mean_length",
                    "completions/clipped_ratio", "reward", "rewards/reward_corr/mean"]
            snap = {k: state.log_history[-1].get(k) for k in keys if state.log_history
                    and k in state.log_history[-1]}
            RECEIPT.setdefault("per_step", []).append({"step": state.global_step, "fields": snap})
            flush()
            marker("step", "done", step=state.global_step,
                   keys=",".join(k for k in snap if snap[k] is not None))

    trainer.add_callback(StepDump())
    t1 = time.time()
    marker("train", "start", steps=STEPS)
    try:
        trainer.train()
        RECEIPT["train_status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        RECEIPT["train_status"] = f"error: {type(exc).__name__}: {exc}"
        RECEIPT["train_traceback_tail"] = traceback.format_exc()[-2000:]
        flush()
        marker("train", "ERROR", err=str(exc)[:200])
        (OUT / "rewards_trace.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in TRACE), encoding="utf-8")
        return 1
    RECEIPT["train_seconds"] = round(time.time() - t1, 1)
    RECEIPT["log_history"] = trainer.state.log_history
    flush()
    marker("train", "done", seconds=RECEIPT["train_seconds"])

    # tier statistics from the real trace
    tiers: dict[str, int] = {}
    rewards = []
    for t in TRACE:
        tiers[t["tier"]] = tiers.get(t["tier"], 0) + 1
        rewards.append(t["reward"])
    RECEIPT["tier_counts"] = tiers
    RECEIPT["n_completions_graded"] = len(TRACE)
    RECEIPT["reward_mean"] = round(sum(rewards) / max(1, len(rewards)), 4) if rewards else None
    if rewards:
        mean = sum(rewards) / len(rewards)
        var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
        RECEIPT["reward_std_overall"] = round(math.sqrt(var), 4)
    flush()
    (OUT / "rewards_trace.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in TRACE), encoding="utf-8")
    print("**R4-1_SUMMARY** " + json.dumps({k: RECEIPT.get(k) for k in (
        "tier_counts", "n_completions_graded", "reward_mean", "train_status",
        "grpo_config_fields_used")}, ensure_ascii=False), flush=True)
    marker("receipt", "done", path=str(OUT / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
