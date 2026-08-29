"""R4-2: micro-batch scan (4.3) — freeze the P3 micro value.

Two runs, both effective batch 16 (same optimization trajectory in theory):
  A: micro=8,  accum=2  (cee4112f baseline config)
  B: micro=16, accum=1  (headroom candidate)
micro=12 is dropped: 16/12 is not an integer, so it cannot hold effective
batch 16 and is not commensurate (recorded in the receipt).

Measured per run: wall seconds, s/step, peak VRAM (max_memory_allocated),
grad_norm/loss/entropy trajectories from trainer_state, tokens/s.
150 steps each, lr 2e-5, same data as cee4112f for comparability.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

VERSION = "r4-2-microbatch-scan-v1"
BASE_MODEL = "/data/magnus/models/Qwen3.5-9B-20260828"
DATA = "/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl"
OUT = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r4-2-microbatch-v1"))
MAX_STEPS = 150
LR = 2e-5
MAX_LEN = 4096

RECEIPT: dict[str, Any] = {"version": VERSION}


def marker(stage: str, status: str, **kw: Any) -> None:
    print(f"=== R4-2 {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def flush() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "receipt.json")


def run_case(key: str, micro: int, accum: int, tok, ds) -> dict[str, Any]:
    import dataclasses
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, get_peft_model

    case_out = OUT / key
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False, attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(r=32, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05)
    m = get_peft_model(model, lora)
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw: dict[str, Any] = dict(
        output_dir=str(case_out),
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=10000,
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
        kw["warmup_steps"] = 8
    if "lr_scheduler_type" in fields:
        kw["lr_scheduler_type"] = "cosine"
    elif "lr_scheduler" in fields:
        kw["lr_scheduler"] = "cosine"
    cfg = SFTConfig(**kw)
    trainer = SFTTrainer(model=m, args=cfg, train_dataset=ds, processing_class=tok)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    marker(key, "start", micro=micro, accum=accum, eff=micro * accum)
    trainer.train()
    wall = time.time() - t0
    peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    hist = trainer.state.log_history
    traj = [{"step": h.get("step"), "loss": h.get("loss"), "grad_norm": h.get("grad_norm"),
             "entropy": h.get("entropy"), "lr": h.get("learning_rate")}
            for h in hist if "loss" in h]
    res = {"micro": micro, "accum": accum, "eff_batch": micro * accum,
           "wall_s": round(wall, 1), "s_per_step": round(wall / MAX_STEPS, 2),
           "peak_vram_gb": peak_gb,
           "tokens_total": hist[-1].get("num_tokens") if hist else None,
           "final_loss": traj[-1]["loss"] if traj else None,
           "grad_norm_max": max((t["grad_norm"] or 0) for t in traj) if traj else None,
           "trajectory": traj}
    RECEIPT["cases"][key] = res
    flush()
    marker(key, "done", s_per_step=res["s_per_step"], peak_gb=peak_gb)
    del m, trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> int:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    flush()
    marker("env", "done", torch=torch.__version__)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False, attn_implementation="sdpa")
    model.config.use_cache = False
    RECEIPT["load_s"] = round(time.time() - t0, 1)
    flush()
    marker("model_load", "done", seconds=RECEIPT["load_s"])

    rows = []
    with Path(DATA).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("protocol_valid") is True:
                rows.append({"messages": row["messages"],
                             "chat_template_kwargs": {"enable_thinking": False}})
    ds = Dataset.from_list(rows)
    RECEIPT["rows"] = len(ds)
    flush()
    marker("dataset", "done", rows=len(ds))

    RECEIPT["cases"] = {}
    run_case("micro8_accum2", 8, 2, tok, ds)
    run_case("micro16_accum1", 16, 1, tok, ds)

    a, b = RECEIPT["cases"]["micro8_accum2"], RECEIPT["cases"]["micro16_accum1"]
    RECEIPT["comparison"] = {
        "speedup": round(a["s_per_step"] / max(0.01, b["s_per_step"]), 3),
        "vram_a": a["peak_vram_gb"], "vram_b": b["peak_vram_gb"],
        "grad_norm_max_a": a["grad_norm_max"], "grad_norm_max_b": b["grad_norm_max"],
    }
    flush()
    print("**R4-2_SUMMARY** " + json.dumps(RECEIPT["comparison"], ensure_ascii=False), flush=True)
    marker("receipt", "done", path=str(OUT / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
