"""150-step, 500-row Qwen3.5-9B SFT speed matrix probe.

The probe is intentionally independent of formal P3 outputs.  VARIANT selects
one controlled change: baseline, no_ckpt, group_length, liger, compile, or ddp.
It records wall time per optimizer step, peak VRAM, and grad norm trajectory.
"""
from __future__ import annotations

import json, os, time, traceback
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

VARIANT = os.environ.get("SPEED_VARIANT", "baseline")
MODEL = os.environ.get("SPEED_MODEL", "/data/magnus/models/Qwen3.5-9B-20260828")
DATA = os.environ.get("SPEED_DATA", "/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl")
OUT = Path(os.environ.get("SPEED_OUT", f"/data/magnus/closedloop-0828/sft-speed/{VARIANT}"))
MAX_LEN = 4096
MAX_STEPS = 150


def rows500():
    rows = []
    with Path(DATA).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                rows.append({"messages": r["messages"],
                             "chat_template_kwargs": {"enable_thinking": False}})
                if len(rows) >= 500:
                    break
    if len(rows) < 500:
        raise RuntimeError(f"only {len(rows)} valid rows")
    return rows


class ProbeCallback(TrainerCallback):
    def __init__(self):
        self.last_step_t = None
        self.step_times = []
        self.grad = []
        self._sync = None

    def on_step_begin(self, args, state, control, **kw):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self._sync = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if self._sync is not None and state.global_step > 0:
            self.step_times.append(time.perf_counter() - self._sync)
        hist = [x for x in state.log_history if "grad_norm" in x]
        if hist: self.grad.append(float(hist[-1]["grad_norm"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = rows500()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_safetensors=True, trust_remote_code=False,
        attn_implementation="sdpa")
    model.config.use_cache = False
    for n, p in model.named_parameters():
        if any(x in n.lower() for x in ("vision", "visual", "image_processor", "merger")):
            p.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32,
        target_modules="all-linear", lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    fields = {f.name for f in __import__("dataclasses").fields(SFTConfig)}
    kw = dict(output_dir=str(OUT / "trainer"), per_device_train_batch_size=8,
              gradient_accumulation_steps=2, max_steps=MAX_STEPS,
              learning_rate=2e-5, weight_decay=0.0, max_grad_norm=1.0,
              bf16=True, gradient_checkpointing=(VARIANT != "no_ckpt"),
              logging_steps=10, save_strategy="no", max_length=MAX_LEN,
              packing=False, dataset_text_field="text", assistant_only_loss=True,
              report_to=[], seed=20260830)
    if "group_by_length" in fields: kw["group_by_length"] = VARIANT == "group_length"
    if "ddp_find_unused_parameters" in fields: kw["ddp_find_unused_parameters"] = False
    if VARIANT == "liger":
        if "use_liger_loss" in fields: kw["use_liger_loss"] = True
        elif "use_liger" in fields: kw["use_liger"] = True
    if "warmup_ratio" in fields: kw["warmup_ratio"] = 0.05
    if "lr_scheduler_type" in fields: kw["lr_scheduler_type"] = "cosine"
    cfg = SFTConfig(**{k:v for k,v in kw.items() if k in fields})
    if VARIANT == "compile":
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    cb = ProbeCallback()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows),
                         processing_class=tok, callbacks=[cb])
    t0 = time.time(); status = "success"; err = None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train()
    except Exception as e:
        status = "error"; err = f"{type(e).__name__}: {e}"; traceback.print_exc()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated()/(1024**3) if torch.cuda.is_available() else None
    times = cb.step_times[-140:]  # discard compile/warm-up tail only if present
    receipt = {"variant": VARIANT, "status": status, "error": err,
      "rows": len(rows), "max_steps": MAX_STEPS, "world_size": int(os.environ.get("WORLD_SIZE", "1")),
      "torch": torch.__version__, "cuda": torch.version.cuda,
      "wall_s": round(wall, 2), "step_s_mean": round(sum(times)/len(times), 3) if times else None,
      "step_s_p50": round(sorted(times)[len(times)//2], 3) if times else None,
      "peak_vram_gib": round(peak, 3) if peak is not None else None,
      "grad_norm_first10": cb.grad[:10], "grad_norm_last10": cb.grad[-10:],
      "config": kw}
    (OUT/"receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SPEED_MATRIX_RECEIPT", json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
