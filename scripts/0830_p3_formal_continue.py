"""P3 formal SFT (20k x 2ep) — pre-registered observables + kill-switches.

Implements 03a as frozen by 05 review:
  - data: 19,800 new trajectories + 1,500 replay rows (alpaca-cleaned, ~7%),
    assistant-only loss, packing false, max_len 4096;
  - hyperparams: lr 2e-5, r32/a32 all-linear, eff batch 16 (micro frozen by
    R4-2), cosine + warmup 5%, wd 0, max_grad_norm 1.0;
  - every 500 steps from step 500: dev-mini dual-column (format + sym-equiv,
    50 questions, multi-turn loop, real sandbox) ->书面判读 fields;
  - dual checkpoint rule (F3): eval-arm = lexicographic best; P4-entry =
    earliest checkpoint with format>=60% AND sym>0; all checkpoints kept;
  - kill-switches: K1 dev-format <40% at two consecutive checkpoints;
    K2 any NaN/inf or loss==0; K3 soft: train loss<0.2 and dev flat 3
    consecutive checkpoints -> stop (memorization);
  - forget probe (F6): held-out replay 50 rows, eval-loss ratio vs base
    measured once at start; >1.10 at a checkpoint -> I-08 action flag;
  - F5 notification: one line per checkpoint with K status + relative link.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

VERSION = "p3-formal-continue-v2"
BASE_MODEL = os.environ.get("P3_MODEL", "/data/magnus/models/Qwen3.5-9B-20260828")
TRAIN_DATA = os.environ.get("P3_DATA", "/data/magnus/closedloop-0828/p2-20k/sft_trajectories.jsonl")
REPLAY_DATA = os.environ.get("P3_REPLAY", "/data/magnus/closedloop-0828/r4-3-f6-freeze-v1/replay-1500.jsonl")
PROBE_DATA = os.environ.get("P3_PROBE", "/data/magnus/closedloop-0828/r4-3-f6-freeze-v1/forget-probe-heldout-50.jsonl")
PROBLEMS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
OUT = Path(os.environ.get("P3_OUT", "/data/magnus/closedloop-0828/p3-formal-continue-v2"))
RESUME = os.environ.get("P3_RESUME", "/data/magnus/closedloop-0828/p3-formal-v1/trainer/checkpoint-1000")
SYSTEM_SHA = "8ed1122a47ae089b1f577d61ad906cf4f7aa5f39627bfef7b6bf2afe79be3217"
MICRO = int(os.environ.get("P3_MICRO", "8"))
ACCUM = int(os.environ.get("P3_ACCUM", "2"))
EPOCHS = 2
LR = 2e-5
MAX_LEN = 4096
DEV_MINI = 50
EVAL_EVERY = 500
FORMAT_GATE = 0.40          # K1
FORMAT_TARGET = 0.60        # F3 P4-entry threshold
K3_LOSS = 0.2
K3_FLAT_RUN = 3
PROBE_RATIO_LIMIT = 1.10    # I-08

RECEIPT: dict[str, Any] = {"version": VERSION, "kill_log": []}
SPECIAL = re.compile(r"<\|[^<>]*\|>")


def marker(stage: str, status: str, **kw: Any) -> None:
    print(f"=== P3F {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def notify(msg: str) -> None:
    """F5 checkpoint-boundary notification for the planning agent."""
    line = f"*** P3F-NOTIFY *** {msg}"
    print(line, flush=True)


def klog(kill: str, detail: str) -> None:
    RECEIPT["kill_log"].append({"kill": kill, "detail": detail, "t": time.time()})
    flush()
    emit("p3f.kill", 1, kind="counter", labels={"kill": kill})
    notify(f"KILL-SWITCH {kill}: {detail}")


def flush() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "receipt.json")




def emit(name: str, value: float, *, kind: str = "gauge", step: int | None = None,
         step_domain: str | None = None, unit: str | None = None,
         labels: dict[str, str] | None = None) -> bool:
    """Magnus Metrics Protocol v1, fail-open (mirrors public/mathphys/metrics.py)."""
    try:
        value = float(value)
        import math as _m
        if not _m.isfinite(value) or kind not in {"gauge", "counter"}:
            return False
        directory = os.environ.get("MAGNUS_METRICS_DIR")
        if not directory or not os.path.isdir(directory):
            return False
        point: dict[str, Any] = {"name": name, "kind": kind, "value": value,
                                 "time_unix_ms": int(time.time() * 1000)}
        if step is not None:
            point["step"] = int(step)
            point["step_domain"] = step_domain or "global"
        if unit is not None:
            point["unit"] = unit
        if labels:
            point["labels"] = {str(k): str(v) for k, v in labels.items()}
        with (Path(directory) / "rank-0.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(point, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
        return True
    except Exception:
        return False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_format(text: str) -> bool:
    cleaned = SPECIAL.sub("", text)
    tags = [m.group(0).lower() for m in re.finditer(r"</?[^<>\s]+>", cleaned)]
    allowed = {"<run>", "</run>", "<final>", "</final>"}
    if any(t not in allowed for t in tags):
        return False
    runs = re.findall(r"<run>\s*.*?\s*</run>", cleaned, flags=re.S)
    finals = re.findall(r"<final>\s*.*?\s*</final>", cleaned, flags=re.S)
    if not runs or len(finals) != 1:
        return False
    expected: list[str] = []
    for _ in runs:
        expected += ["<run>", "</run>"]
    expected += ["<final>", "</final>"]
    if tags != expected:
        return False
    first = re.search(r"</?[^<>\s]+>", cleaned)
    if cleaned[:first.start()].strip():
        return False
    return not cleaned[re.search(r"</final>", cleaned).end():].strip()


def sym_equiv(final_text: str, gold: str) -> bool:
    try:
        import sympy as sym
        got = sym.sympify(final_text.strip())
        want = sym.sympify(gold)
        if sym.simplify(got - want) == 0:
            return True
        f1, f2 = sym.lambdify([], got, "numpy"), sym.lambdify([], want, "numpy")
        return abs(float(f1()) - float(f2())) < 1e-9
    except Exception:  # noqa: BLE001
        return False


def sandbox_exec(code: str) -> dict[str, Any]:
    prelude = "import numpy as np\nimport scipy as sp\nimport sympy as sym\n"
    try:
        proc = subprocess.run(["python3", "-c", prelude + code], capture_output=True,
                              text=True, timeout=10.0)
        return {"ok": proc.returncode == 0,
                "output": (proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout))[:8000]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "<timeout>"}


def align(prompt: str) -> str:
    if prompt.endswith("<think>\n"):
        return prompt + "\n</think>\n\n"
    if prompt.endswith("<|im_start|>assistant\n"):
        return prompt + "<think>\n\n</think>\n\n"
    return prompt


def render_context(tok: Any, system: str, user: str) -> str:
    return align(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False}))


def dev_mini_dual(model: Any, tok: Any, system: str, questions: list[dict]) -> dict[str, Any]:
    """Multi-turn loop, real sandbox, format + CAS columns (verifier B-side
    proxy: direct canonical compare; full B-face verifier ships with P4)."""
    import torch
    n_fmt = n_sym = n_total = 0
    rows = []
    model.eval()
    for q in questions[:DEV_MINI]:
        prompt = render_context(tok, system, q["prompt"])
        transcript = ""
        cur = prompt
        for _ in range(3):
            ids = tok(cur, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            w = ids["input_ids"].shape[1]
            with torch.no_grad():
                seq = model.generate(**ids, max_new_tokens=1024, do_sample=False,
                                     pad_token_id=tok.pad_token_id)[0]
            text = tok.decode(seq[w:], skip_special_tokens=False)
            transcript += text
            run_m = re.search(r"<run>\s*(.*?)\s*</run>", text, flags=re.S)
            final_m = re.search(r"<final>\s*(.*?)\s*</final>", text, flags=re.S)
            if final_m and not run_m:
                break
            if run_m:
                ex = sandbox_exec(run_m.group(1))
                cur = (cur + SPECIAL.split(text)[0] + "<|im_end|>\n<|im_start|>user\n"
                       + f"<output>\n{ex['output']}\n</output><|im_end|>\n"
                       + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            else:
                break
        fmt = strict_format(transcript)
        fm = re.search(r"<final>\s*(.*?)\s*</final>", transcript, flags=re.S)
        sym_ok = bool(fm) and sym_equiv(fm.group(1), q["gold_canonical"])
        n_fmt += fmt
        n_sym += sym_ok
        n_total += 1
        rows.append({"problem_id": q["id"], "format": fmt, "sym": sym_ok})
    model.train()
    return {"n": n_total, "format_rate": round(n_fmt / max(1, n_total), 4),
            "sym_rate": round(n_sym / max(1, n_total), 4), "rows": rows[:5]}


def probe_loss(model: Any, tok: Any, rows: list[dict]) -> float:
    import torch
    from torch.nn import CrossEntropyLoss
    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows[:50]:
            msgs = []
            if row.get("instruction"):
                user = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
                msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": row.get("output", "")}]
            else:
                msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                text = tok.apply_chat_template(msgs, tokenize=False)
                ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                labels = ids["input_ids"].clone()
                out = model(**ids, labels=labels)
                losses.append(float(out.loss))
            except Exception:  # noqa: BLE001
                continue
    model.train()
    return round(sum(losses) / max(1, len(losses)), 4)


class KillSwitch(Exception):
    pass


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
    venv = Path("/dev/shm/r4-1c/venv")
    py = venv / "bin" / "python"
    marker("bootstrap", "start", reason="FSDPModule missing in torch 2.5.1; v3 adds torchvision 0.22.1 ABI match")
    try:
        free = shutil.disk_usage("/dev/shm").free
    except Exception:
        free = 0
    if free < 12 * 2**30:
        venv = Path("/tmp/r4-1c/venv")
        py = venv / "bin" / "python"
    env = dict(os.environ)
    env["PIP_NO_COMPILE"] = "1"
    if not py.exists():
        subprocess.run(["python3", "-m", "venv", "--system-site-packages", str(venv)],
                       check=True, timeout=300)
        wheels = Path("/tmp/r4-1c/wheels")
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
    from datasets import Dataset, concatenate_datasets
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    RECEIPT["metrics_dir_present"] = bool(os.environ.get("MAGNUS_METRICS_DIR"))
    flush()
    marker("env", "done", torch=torch.__version__, micro=MICRO, accum=ACCUM,
           metrics_dir=RECEIPT["metrics_dir_present"])
    emit("p3f.env", 1, kind="counter", labels={"micro": str(MICRO), "accum": str(ACCUM)})

    # data
    train_rows = []
    with Path(TRAIN_DATA).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                train_rows.append({"messages": r["messages"],
                                   "chat_template_kwargs": {"enable_thinking": False}})
    replay_rows = []
    with Path(REPLAY_DATA).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r.get("instruction", "")
            if r.get("input"):
                user += "\n" + r["input"]
            if user and r.get("output"):
                replay_rows.append({"messages": [{"role": "user", "content": user},
                                                 {"role": "assistant", "content": r["output"]}],
                                    "chat_template_kwargs": {"enable_thinking": False}})
    probe_rows = [json.loads(l) for l in Path(PROBE_DATA).open(encoding="utf-8") if l.strip()]
    RECEIPT["data"] = {"train_rows": len(train_rows), "replay_rows": len(replay_rows),
                       "replay_frac": round(len(replay_rows) / max(1, len(train_rows) + len(replay_rows)), 4),
                       "train_sha256": sha256_file(TRAIN_DATA),
                       "replay_sha256": sha256_file(REPLAY_DATA),
                       "probe_rows": len(probe_rows)}
    flush()
    marker("data", "done", train=len(train_rows), replay=len(replay_rows),
           frac=RECEIPT["data"]["replay_frac"])

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False, attn_implementation="sdpa")
    model.config.use_cache = False
    RECEIPT["load_s"] = round(time.time() - t0, 1)
    flush()
    marker("model_load", "done", seconds=RECEIPT["load_s"])

    # system prompt + dev-mini questions
    system = None
    with Path(TRAIN_DATA).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                system = r["messages"][0]["content"]
                break
    RECEIPT["system_sha_matches"] = hashlib.sha256(system.encode()).hexdigest() == SYSTEM_SHA
    questions = []
    with Path(PROBLEMS).open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q.get("split") in {"dev", "eval", "test", "holdout_family"}:
                gold = q["gold"]["canonical_sympy"] if isinstance(q.get("gold"), dict) else str(q.get("gold", ""))
                questions.append({"id": q["id"], "prompt": q["prompt"], "gold_canonical": gold})
    flush()
    marker("prompts", "done", sys_ok=RECEIPT["system_sha_matches"], dev=len(questions))

    # I-08 baseline: probe loss of the base model
    base_probe = probe_loss(model, tok, probe_rows)
    RECEIPT["probe_loss_base"] = base_probe
    flush()
    marker("probe_baseline", "done", loss=base_probe)

    for name, param in model.named_parameters():
        if any(part in name.lower() for part in ("vision", "visual", "image_processor", "merger")):
            param.requires_grad = False
    lora = LoraConfig(r=32, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05)
    model = get_peft_model(model, lora)
    RECEIPT["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flush()
    marker("lora", "done", trainable=RECEIPT["trainable"])

    ds = Dataset.from_list(train_rows + replay_rows)
    import dataclasses
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw: dict[str, Any] = dict(
        output_dir=str(OUT / "trainer"),
        per_device_train_batch_size=MICRO,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_only_model=False,
        max_length=MAX_LEN,
        packing=False,
        dataset_text_field="text",
        assistant_only_loss=True,
        report_to=[],
        seed=20260830,
    )
    if "warmup_ratio" in fields:
        kw["warmup_ratio"] = 0.05
    elif "warmup_steps" in fields:
        kw["warmup_steps"] = 125
    if "lr_scheduler_type" in fields:
        kw["lr_scheduler_type"] = "cosine"
    elif "lr_scheduler" in fields:
        kw["lr_scheduler"] = "cosine"
    RECEIPT["sft_config_fields_used"] = sorted(kw.keys())
    flush()
    cfg = SFTConfig(**kw)

    state = {"last_dev_fmt": None, "flat_streak": 0, "best": None, "p4_entry": None,
             "prev_ckpt_dir": None, "step": 0}

    def evaluate_and_decide(trainer, step: int) -> None:
        dual = dev_mini_dual(trainer.model, tok, system, questions)
        train_hist = [h for h in trainer.state.log_history if "loss" in h]
        recent_loss = train_hist[-1]["loss"] if train_hist else None
        nan_hit = any((h.get("loss") is None) or math.isnan(h["loss"]) or math.isinf(h["loss"])
                      for h in train_hist)
        probe = probe_loss(trainer.model, tok, probe_rows)
        probe_ratio = round(probe / base_probe, 4) if base_probe else None

        ck = {"step": step, "dev": dual, "recent_loss": recent_loss,
              "probe_loss": probe, "probe_ratio": probe_ratio}
        RECEIPT.setdefault("checkpoints", []).append(ck)
        emit("p3f.dev.format_rate", dual["format_rate"], step=step)
        emit("p3f.dev.sym_rate", dual["sym_rate"], step=step)
        if probe_ratio is not None:
            emit("p3f.probe_ratio", probe_ratio, step=step)
        emit("p3f.checkpoint", 1, kind="counter", step=step)
        # F3 dual selection
        qualifies = dual["format_rate"] >= FORMAT_TARGET and dual["sym_rate"] > 0
        if qualifies and (state["p4_entry"] is None):
            state["p4_entry"] = step
        best = state["best"]
        better = (best is None or (dual["format_rate"], dual["sym_rate"]) >
                  (best["dev"]["format_rate"], best["dev"]["sym_rate"]))
        if better:
            state["best"] = ck
        # K2
        if nan_hit or (recent_loss is not None and recent_loss == 0.0):
            klog("K2", f"NaN/inf or zero loss at step {step}")
            raise KillSwitch("K2")
        # K1/K3 in OBSERVATION MODE (user 2026-08-30: keep running to capture the
        # full overfitting curve for the group-meeting report): log would-fire,
        # never raise. K2 (NaN) remains a hard stop - that is breakage, not data.
        if dual["format_rate"] < FORMAT_GATE:
            if state["last_dev_fmt"] is not None and state["last_dev_fmt"] < FORMAT_GATE:
                klog("K1-would-fire", f"dev format {dual['format_rate']} < {FORMAT_GATE} twice (obs mode, continue)")
                emit("p3f.dev.k1_would_fire", 1, kind="counter", step=step)
            state["last_dev_fmt"] = dual["format_rate"]
        else:
            state["last_dev_fmt"] = dual["format_rate"]
        if recent_loss is not None and recent_loss < K3_LOSS:
            if dual == state.get("prev_dual"):
                state["flat_streak"] += 1
            else:
                state["flat_streak"] = 0
            state["prev_dual"] = dual
        # I-08 flag
        if probe_ratio and probe_ratio > PROBE_RATIO_LIMIT:
            RECEIPT.setdefault("probe_breaches", []).append({"step": step, "ratio": probe_ratio})
            notify(f"I-08 probe_ratio={probe_ratio} > {PROBE_RATIO_LIMIT} at step {step} (action: replay 10%/epoch 1)")
        k_status = {k: "ok" for k in ("K1", "K2", "K3")}
        for entry in RECEIPT["kill_log"]:
            k_status[entry["kill"]] = "TRIGGERED"
        notify(f"checkpoint step={step} dev_format={dual['format_rate']} dev_sym={dual['sym_rate']} "
               f"loss={recent_loss} probe_ratio={probe_ratio} K={json.dumps(k_status)} "
               f"best={state['best']['step'] if state['best'] else None} "
               f"p4_entry={state['p4_entry']} frontend=/jobs/<current> (receipt {OUT/'receipt.json'})")
        flush()

    class GateCallback(TrainerCallback):
        last_emitted_log_step = None

        def on_step_end(self, args, state, control, **kw2):
            hist = [h for h in state.log_history if "loss" in h]
            # Emit ONLY when trainer logging produces a new point (logging_steps);
            # re-emitting the stale value every step renders as stair-steps.
            if hist and hist[-1].get("step") != self.last_emitted_log_step:
                h = hist[-1]
                self.last_emitted_log_step = h.get("step")
                for key, name in (("loss", "p3f.train.loss"), ("grad_norm", "p3f.train.grad_norm"),
                                  ("learning_rate", "p3f.train.lr"), ("entropy", "p3f.train.entropy"),
                                  ("mean_token_accuracy", "p3f.train.token_acc")):
                    if h.get(key) is not None:
                        emit(name, h[key], step=h["step"])
            if state.global_step % EVAL_EVERY == 0:
                evaluate_and_decide(trainer_ref[0], state.global_step)

    trainer_ref = [None]
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer_ref[0] = trainer
    trainer.add_callback(GateCallback())

    t1 = time.time()
    if not Path(RESUME).exists():
        # NEVER silently fresh-run a continuation (R4 lesson: v2c burned a GPU
        # for 15 min because of this fallback)
        raise RuntimeError(f"resume checkpoint missing: {RESUME}")
    marker("train", "start", resume_from=RESUME)
    status = "completed"
    try:
        trainer.train(resume_from_checkpoint=RESUME)
    except KillSwitch as ks:
        status = f"kill-switch {ks}"
        marker("train", "KILLED", reason=str(ks))
    except Exception as exc:  # noqa: BLE001
        status = f"error: {type(exc).__name__}: {exc}"
        RECEIPT["train_traceback_tail"] = traceback.format_exc()[-2000:]
        marker("train_error", "hit", err=str(exc)[:200])
    RECEIPT["train_status"] = status
    RECEIPT["train_seconds"] = round(time.time() - t1, 1)
    RECEIPT["log_history_tail"] = trainer.state.log_history[-10:]
    flush()
    marker("train", "done", seconds=RECEIPT["train_seconds"], status=status)

    # merge + dual artifact save
    marker("merge", "start")
    merged = model.merge_and_unload()
    (OUT / "merged-final").mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUT / "merged-final"), safe_serialization=True)
    tok.save_pretrained(str(OUT / "merged-final"))
    RECEIPT["artifacts"] = {
        "eval_arm_rule": "dual-column lexicographic best checkpoint",
        "final_dir": str(OUT / "merged-final"),
        "v1_best_step": state["best"]["step"] if state["best"] else None,
        "p4_entry_rule": f"earliest checkpoint with format>={FORMAT_TARGET} and sym>0",
        "p4_entry_step": state["p4_entry"],
    }
    flush()
    marker("merge", "done", eval_arm=RECEIPT["artifacts"]["eval_arm_step"],
           p4_entry=RECEIPT["artifacts"]["p4_entry_step"])

    notify(f"P3 FINISHED status={status} train_seconds={RECEIPT['train_seconds']} "
           f"eval_arm_step={RECEIPT['artifacts']['eval_arm_step']} "
           f"p4_entry_step={RECEIPT['artifacts']['p4_entry_step']}")
    print("**P3F_SUMMARY** " + json.dumps({k: RECEIPT.get(k) for k in (
        "train_status", "train_seconds", "checkpoints", "artifacts", "kill_log")},
        ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
