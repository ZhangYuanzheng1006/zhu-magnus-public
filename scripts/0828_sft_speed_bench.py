"""Benchmark Qwen3.5-9B LoRA training throughput on one A100.

This is an isolated speed probe. It does not modify the formal P3 job or its
outputs. Each configuration uses the same eight tokenized P2 trajectories and
one effective optimizer step, then reports wall time and peak CUDA memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/magnus/models/Qwen3.5-9B-20260828")
    p.add_argument("--data", default="/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl")
    p.add_argument("--out", default="/data/magnus/closedloop-0828/sft-speed-bench")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def read_rows(path: str, count: int = 16):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("protocol_valid") is True:
                rows.append(row)
                if len(rows) >= count:
                    break
    if len(rows) < 8:
        raise RuntimeError(f"need at least 8 valid rows, found {len(rows)}")
    return rows


def make_batch(tok, rows, max_length: int):
    encoded = []
    for row in rows[:8]:
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        item = tok(text, truncation=True, max_length=max_length, add_special_tokens=False)
        encoded.append(item["input_ids"])
    width = max(len(x) for x in encoded)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(encoded):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, :len(seq)] = 1
    return ids.cuda(), mask.cuda(), int(mask.sum().item())


def run_config(model, ids, mask, tokens, micro_batch: int, grad_accum: int, checkpointing: bool, warmup: int):
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1.5e-4)
    total_rows = ids.shape[0]
    if micro_batch * grad_accum > total_rows:
        raise ValueError("benchmark config exceeds fixed batch rows")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        for k in range(grad_accum):
            lo = k * micro_batch
            hi = lo + micro_batch
            out = model(input_ids=ids[lo:hi], attention_mask=mask[lo:hi], labels=ids[lo:hi])
            (out.loss / grad_accum).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    for k in range(grad_accum):
        lo = k * micro_batch
        hi = lo + micro_batch
        out = model(input_ids=ids[lo:hi], attention_mask=mask[lo:hi], labels=ids[lo:hi])
        (out.loss / grad_accum).backward()
    torch.cuda.synchronize()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    result = {
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "effective_samples": micro_batch * grad_accum,
        "checkpointing": checkpointing,
        "elapsed_s": round(elapsed, 4),
        "tokens": tokens,
        "tokens_per_s": round(tokens / elapsed, 2),
        "samples_per_s": round((micro_batch * grad_accum) / elapsed, 4),
        "peak_memory_gib": round(peak, 3),
        "loss": float(out.loss.detach().cpu()),
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer, out
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = read_rows(args.data)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    ids, mask, tokens = make_batch(tok, rows, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    configs = [
        (2, 4, True),
        (4, 2, True),
        (8, 1, True),
        (4, 2, False),
    ]
    results = []
    for micro, accum, ckpt in configs:
        print(f"=== config micro={micro} accum={accum} checkpointing={ckpt} ===", flush=True)
        try:
            result = run_config(model, ids, mask, tokens, micro, accum, ckpt, args.warmup)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            result = {"micro_batch": micro, "grad_accum": accum, "checkpointing": ckpt, "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            gc.collect(); torch.cuda.empty_cache()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "model": args.model,
        "data": args.data,
        "rows": len(rows),
        "max_length": args.max_length,
        "tokens_in_fixed_batch": tokens,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
        "recommendation": "choose the fastest non-OOM configuration after comparing results; this probe does not alter P3",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
