"""
Qwen3.5-9B TRL SFT + LoRA smoke for the CUDA 12.4 training image.

The runtime deliberately has no vLLM dependency.  Training and merged-model
checks use Transformers; teacher/vLLM deployment is a separate job because
current vLLM releases pin CUDA 13 Torch versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from transformers import TrainerCallback
from runtime_config import STUDENT

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


METRIC_FILE = "p3-student.jsonl"


def emit_metric(
    name: str,
    value: float,
    *,
    kind: str = "gauge",
    step: int | None = None,
    step_domain: str | None = None,
    unit: str | None = None,
    labels: dict[str, str] | None = None,
) -> bool:
    """Fail-open scalar Metrics Protocol v1 emitter."""
    try:
        value = float(value)
        if not math.isfinite(value) or kind not in {"gauge", "counter"}:
            return False
        directory = os.environ.get("MAGNUS_METRICS_DIR")
        if not directory or not os.path.isdir(directory):
            return False
        point: dict[str, object] = {
            "name": name,
            "kind": kind,
            "value": value,
            "time_unix_ms": int(time.time() * 1000),
        }
        if step is not None:
            point["step"] = int(step)
            point["step_domain"] = step_domain or "optimizer"
        if unit is not None:
            point["unit"] = unit
        if labels:
            point["labels"] = {str(k): str(v) for k, v in labels.items()}
        with (Path(directory) / METRIC_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(point, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("P3_MODEL", "/data/magnus/models/Qwen3.5-9B-20260828"))
    p.add_argument("--data", default=os.environ.get("P3_DATA", "/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl"))
    p.add_argument("--out", default=os.environ.get("P3_OUT", "/data/magnus/models/Qwen3.5-9B-sft-20260828"))
    p.add_argument("--lr", type=float, default=float(os.environ.get("P3_LR", "1.5e-4")))
    p.add_argument("--r", type=int, default=int(os.environ.get("P3_LORA_R", "16")))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("P3_MAX_STEPS", "150")))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("P3_MAX_SEQ_LEN", str(STUDENT["sft_max_seq_len"]))))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("P3_BATCH_SIZE", "8")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("P3_GRAD_ACCUM", "1")))
    p.add_argument("--require-kernels", action="store_true", default=os.environ.get("P3_REQUIRE_KERNELS", "0") == "1")
    p.add_argument("--scheduler", default=os.environ.get("P3_SCHEDULER", "constant"))
    p.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("P3_WARMUP_RATIO", "0")))
    p.add_argument("--weight-decay", type=float, default=float(os.environ.get("P3_WEIGHT_DECAY", "0")))
    p.add_argument("--max-grad-norm", type=float, default=float(os.environ.get("P3_MAX_GRAD_NORM", "1.0")))
    p.add_argument("--gradient-checkpointing", action="store_true", default=os.environ.get("P3_GRADIENT_CHECKPOINTING", "0") == "1")
    return p.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_jsonl(src: str, dst: str) -> tuple[int, str]:
    """Filter invalid P2 rows and persist an immutable P3 input snapshot."""
    rows: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("protocol_valid") is True:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no protocol-valid rows in {src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows), sha256_file(dst)


def load_eval_prompts(path: str, limit: int = 50) -> list[dict]:
    """Load a real P1 dev slice when present; never label train rows as dev."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") == "dev":
                rows.append(row)
                if len(rows) >= limit:
                    break
    return rows


def format_hit(text: str) -> bool:
    return all(tag in text for tag in ("<run>", "</run>", "<final>", "</final>"))


def benchmark_checkpoint(
    model,
    tok,
    rows: list[dict],
    step: int,
    system_prompt: str,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> dict:
    """Run a bounded Transformers-only format benchmark on a P1 dev slice."""
    if not rows:
        return {"step": step, "available": False, "reason": "p1 dev slice unavailable"}
    import torch

    was_training = model.training
    model.eval()
    hits = 0
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            texts = []
            for row in batch_rows:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["prompt"]},
                ]
                # The model's tokenizer template is authoritative; P1 context is
                # not a substitute for the approved 06 system prompt.
                if "messages" in row:
                    messages = row["messages"][:2]
                texts.append(tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": STUDENT["enable_thinking"]},
                ))
            inputs = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=STUDENT["max_model_len"],
            ).to(model.device)
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            for row, sequence, prompt_len in zip(batch_rows, generated, prompt_lengths):
                # With right padding, slice by the padded batch width; decode
                # only newly generated tokens for each row.
                answer = tok.decode(sequence[inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                ok = format_hit(answer)
                hits += int(ok)
                outputs.append({"problem_id": row.get("id"), "format_hit": ok, "output": answer[:800]})
    if was_training:
        model.train()
    rate = hits / len(rows)
    emit_metric("eval.checkpoint.format_hit", rate, step=step, step_domain="checkpoint", unit="rate", labels={"phase": "p3"})
    # Candidate semantic equivalence is intentionally unavailable until the
    # verifier parses candidate finals; do not infer it from format alone.
    return {
        "step": step,
        "available": True,
        "count": len(rows),
        "format_hit": hits,
        "format_hit_rate": rate,
        "sym_equiv": None,
        "sym_equiv_reason": "candidate final semantic parser not implemented in verifier v0",
        "samples": outputs[:5],
    }


class TimingCallback(TrainerCallback):
    """Emit Trainer logs, step timing, and checkpoint benchmarks."""

    def __init__(self, tok, dev_rows, system_prompt, checkpoints):
        self.tok = tok
        self.dev_rows = dev_rows
        self.system_prompt = system_prompt
        self.checkpoints = checkpoints
        self.step_started: float | None = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started is not None:
            emit_metric("train.step_time.total", time.perf_counter() - self.step_started, step=state.global_step, unit="s")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = state.global_step
        if "loss" in logs:
            emit_metric("train.loss", logs["loss"], step=step, unit="loss")
        if "learning_rate" in logs:
            emit_metric("train.lr", logs["learning_rate"], step=step, unit="rate")
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.global_step in {75, 150}:
            # Trainer invokes callbacks after the checkpoint is serialized; the
            # in-memory model is the same optimizer state at this point.
            self.checkpoints.append(
                benchmark_checkpoint(self.model, self.tok, self.dev_rows, state.global_step, self.system_prompt)
            )
        return control


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from startup_gate import assert_startup_gate, run_startup_gate, write_gate

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not run P3 on CPU")
    if torch.version.cuda != "12.4":
        raise RuntimeError(f"expected CUDA 12.4 runtime, got {torch.version.cuda}")
    pre_gate = run_startup_gate(attn_implementation="flash_attention_2")
    write_gate(pre_gate, out / "startup_gate-pre-model.json")
    if args.require_kernels:
        assert_startup_gate(pre_gate)

    data_snapshot = out / "p3_input.jsonl"
    count, data_hash = snapshot_jsonl(args.data, data_snapshot)
    print(f"P3 input rows={count} sha256={data_hash}")
    emit_metric("data.rows", count, kind="counter", step=0, step_domain="data", unit="rows", labels={"phase": "p3"})

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=False,
    )
    for name, param in model.named_parameters():
        if any(part in name.lower() for part in ("vision", "visual", "image_processor", "merger")):
            param.requires_grad = False
    post_gate = run_startup_gate(model=model, require_model=True, attn_implementation="flash_attention_2")
    write_gate(post_gate, out / "startup_gate-post-model.json")
    if args.require_kernels:
        assert_startup_gate(post_gate)
    total_params = sum(p.numel() for p in model.parameters())
    lora = LoraConfig(
        r=args.r,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_mod = Counter()
    for name, param in model.named_parameters():
        if param.requires_grad:
            parts = name.split(".")
            by_mod[parts[-2] if len(parts) >= 2 else name] += param.numel()
    parameter_account = {
        "total_params": total_params,
        "trainable_params": trainable,
        "trainable_pct": round(trainable / total_params * 100, 4),
        "trainable_by_module": dict(by_mod),
    }
    print("parameter_account:", json.dumps(parameter_account, ensure_ascii=False))

    ds = load_dataset("json", data_files=str(data_snapshot), split="train")
    print(f"dataset rows: {len(ds)}")

    def add_runtime_options(example):
        return {"chat_template_kwargs": {"enable_thinking": STUDENT["enable_thinking"]}}

    ds = ds.map(add_runtime_options)
    cfg = SFTConfig(
        output_dir=str(out / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_steps=75,
        save_only_model=False,
        max_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
        assistant_only_loss=True,
        report_to=[],
        seed=20260828,
        data_seed=20260828,
    )
    # P1 dev evidence is optional but, when present, is the only source used
    # for checkpoint benchmarking. No semantic score is inferred from format.
    dev_rows = load_eval_prompts("/data/magnus/closedloop-0828/p1/problems.jsonl")
    system_prompt = ds[0]["messages"][0]["content"]
    checkpoints = [benchmark_checkpoint(model, tok, dev_rows, 0, system_prompt)]
    emit_metric("train.step_time.forward", 0.0, step=0, unit="s", labels={"available": "false"})
    emit_metric("train.step_time.backward", 0.0, step=0, unit="s", labels={"available": "false"})
    emit_metric("train.step_time.optimizer", 0.0, step=0, unit="s", labels={"available": "false"})
    callback = TimingCallback(tok, dev_rows, system_prompt, checkpoints)
    callback.model = model
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        data_collator=None,
        callbacks=[callback],
    )
    train_start = time.time()
    result = trainer.train()
    train_seconds = time.time() - train_start

    # The on_save callback records step 150 when save_steps reaches the final
    # step; append a final benchmark only if that callback did not run.
    if not any(item.get("step") == args.max_steps for item in checkpoints):
        checkpoints.append(benchmark_checkpoint(model, tok, dev_rows, args.max_steps, system_prompt))
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    merged = model.merge_and_unload()
    merged_dir = out / "merged"
    merged.save_pretrained(str(merged_dir))
    tok.save_pretrained(str(merged_dir))
    merged_load_ok = False
    merged_load_error = None
    try:
        # Reload on CPU first; this checks the serialized merged artifact
        # without creating a second 9B GPU allocation.
        _ = AutoModelForCausalLM.from_pretrained(str(merged_dir), torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=False)
        merged_load_ok = True
    except Exception as exc:
        merged_load_error = f"{type(exc).__name__}: {exc}"

    last_loss = None
    if result.training_loss is not None:
        last_loss = float(result.training_loss)
    receipt = {
        "args": vars(args),
        "model": args.model,
        "data": str(data_snapshot),
        "data_sha256": data_hash,
        "system_prompt_sha256": hashlib.sha256(ds[0]["messages"][0]["content"].encode("utf-8")).hexdigest(),
        "parameter_account": parameter_account,
        "startup_gate_pre_model": pre_gate,
        "startup_gate_post_model": post_gate,
        "training_config": {
            "lr_scheduler_type": args.scheduler,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "dataset_rows": count,
        "training_seconds": round(train_seconds, 2),
        "max_steps_completed": int(trainer.state.global_step),
        "training_loss": last_loss,
        "checkpoints": checkpoints,
        "adapter_dir": str(out / "adapter"),
        "merged_dir": str(merged_dir),
        "merged_transformers_load_ok": merged_load_ok,
        "merged_transformers_load_error": merged_load_error,
        "vllm_load": {"available": False, "reason": "vLLM intentionally absent from cu124 training image"},
        "total_seconds": round(time.time() - started, 2),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_metric("train.duration", train_seconds, step=trainer.state.global_step, unit="s", labels={"phase": "p3"})
    emit_metric("train.steps_completed", trainer.state.global_step, kind="counter", step=trainer.state.global_step, unit="steps", labels={"phase": "p3"})
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
