"""R3-3: 150-step SFT rerun (07 params) + aligned-prompt format retest.

Root cause from R3-1: the R2-2 eval prompt diverged from the training render at
token 3 (different system prompt) and dropped the model into an open
``<think>`` block it never saw content for. This runtime enforces one shared
context renderer for train and eval:

* ``render_context_prefix`` builds the eval prompt (system + user + generation
  prompt) and appends the empty-think alignment suffix so the model starts
  generating from exactly the assistant-opening distribution seen in training.
* An in-job invariant asserts ``render_context_prefix`` output is a byte-exact
  prefix of the TRL-style full-conversation render, so the two paths provably
  share one construction.

Training keeps the 07 parameter set (lr 2e-5, LoRA r32/alpha32 all-linear,
effective batch 16 = 8 x 2, seq 4096, packing false, cosine, warmup 5 percent,
grad checkpointing on) with TRL's conversational path and assistant-only loss.
After training the adapter is merged, reloaded, and retested on the SAME first
20 questions as R2-2 with strict_parse, saving 5 success + 5 failure raws.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

VERSION = "r3-3-sft-eval-v1"
BASE_MODEL = os.environ.get("R3_MODEL", "/data/magnus/models/Qwen3.5-9B-20260828")
DATA = os.environ.get("R3_DATA", "/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl")
QUESTIONS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
OUT = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r3-3-sft-eval-v1"))
EXPECT_SYSTEM_SHA = "8ed1122a47ae089b1f577d61ad906cf4f7aa5f39627bfef7b6bf2afe79be3217"
MAX_STEPS = 150
LR = 2e-5
LORA_R = 32
LORA_ALPHA = 32
MICRO_BATCH = 8
GRAD_ACCUM = 2
MAX_LEN = 4096
EVAL_LIMIT = 20
MAX_NEW = 4096
EVAL_BATCH = 4

RECEIPT: dict[str, Any] = {"version": VERSION}
TRAIN_MARKER = "=== R3-3"


def marker(stage: str, status: str, **kw: Any) -> None:
    print(f"{TRAIN_MARKER} {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def flush_receipt() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "receipt.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_questions(path: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in {"dev", "eval", "test", "holdout_family"}:
                continue
            row["id"] = str(row.get("id", row.get("problem_id", len(rows))))
            row["prompt"] = str(row.get("prompt", row.get("question", "")))
            if row["prompt"]:
                rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("no eligible questions")
    return rows


def strict_parse(text: str) -> dict[str, Any]:
    """Format-only parser v0; ambiguity, errors, and contradictions abstain."""
    result: dict[str, Any] = {"status": "uncertain", "reason": None}
    if not isinstance(text, str) or not text.strip():
        result["reason"] = "empty_generation"
        return result
    lower = text.lower()
    if any(x in lower for x in ("traceback", "timeoutexpired", "<timeout>", "execution timeout", "resource_exhausted")):
        result["reason"] = "execution_exception_or_timeout"
        return result
    if any(x in text for x in ("无法确定", "不确定", "contradiction", "矛盾", "无法判断")):
        result["reason"] = "explicit_or_detected_contradiction"
        return result
    tag_matches = list(re.finditer(r"</?[^<>\s]+>", text, flags=re.I))
    tags = [m.group(0).lower() for m in tag_matches]
    runs = re.findall(r"<run>\s*(.*?)\s*</run>", text, flags=re.S)
    finals = list(re.finditer(r"<final>\s*(.*?)\s*</final>", text, flags=re.S))
    allowed = {"<run>", "</run>", "<final>", "</final>"}
    if any(tag not in allowed for tag in tags):
        result["reason"] = "unknown_protocol_tag"
        return result
    if not runs:
        result["reason"] = "missing_run"
        return result
    if len(finals) != 1 or not finals[0].group(1).strip():
        result["reason"] = "final_count_or_empty"
        return result
    expected = []
    for _ in runs:
        expected.extend(("<run>", "</run>"))
    expected.extend(("<final>", "</final>"))
    if tags != expected:
        result["reason"] = "tag_order_or_unclosed"
        return result
    first_tag = tag_matches[0]
    if text[:first_tag.start()].strip():
        result["reason"] = "text_before_protocol"
        return result
    tail = text[finals[0].end():].strip()
    if tail:
        result["reason"] = "trailing_text"
        return result
    result.update(status="success", run_count=len(runs), final=finals[0].group(1).strip())
    return result


def load_training_rows() -> list[dict[str, Any]]:
    rows = []
    with Path(DATA).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("protocol_valid") is True:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no protocol-valid rows in {DATA}")
    return rows


def render_context_prefix(tok: Any, system: str, user: str) -> str:
    """Single shared renderer for the eval context (R3-1 root-cause fix).

    Builds system+user+generation-prompt with enable_thinking=False, then
    aligns the assistant opening with the training distribution: the Qwen3.5
    template opens ``<think>`` in the generation prompt, while training always
    shows an EMPTY think block before assistant content, so we close it.
    """
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False})
    if prompt.endswith("<think>\n"):
        prompt += "\n\n</think>\n\n"
    elif prompt.endswith("<|im_start|>assistant\n"):
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def assert_shared_render_invariant(tok: Any, system: str) -> None:
    """Prove eval context == prefix of the TRL-style training render."""
    user = "示例题目:验证 (x+y)^2 展开后与 x^2+2xy+y^2 数值一致。"
    probe = "<run>\nprint(1)\n</run>"
    full = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user},
         {"role": "assistant", "content": probe}],
        tokenize=False, chat_template_kwargs={"enable_thinking": False})
    prefix = render_context_prefix(tok, system, user)
    ok = full.startswith(prefix + probe) or full.startswith(prefix)
    RECEIPT["shared_render_invariant"] = {
        "ok": bool(ok), "prefix_tokens": len(tok(prefix, add_special_tokens=False)["input_ids"]),
        "full_head": full[:160], "prefix_tail": prefix[-80:]}
    flush_receipt()
    marker("shared_render_invariant", "done" if ok else "FAIL")
    if not ok:
        raise RuntimeError("shared render invariant failed: eval context is not a prefix of training render")


def main() -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    flush_receipt()
    marker("env", "done", torch=torch.__version__, cuda=torch.version.cuda)

    rows = load_training_rows()
    system = rows[0]["messages"][0]["content"]
    system_sha = hashlib.sha256(system.encode()).hexdigest()
    RECEIPT["system_prompt_sha256"] = system_sha
    RECEIPT["system_matches_expected"] = system_sha == EXPECT_SYSTEM_SHA
    flush_receipt()
    marker("system_prompt", "done", sha=system_sha[:12], matches=RECEIPT["system_matches_expected"])

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
            device_map="cuda", trust_remote_code=False, attn_implementation="flash_attention_2")
        RECEIPT["attn_implementation"] = "flash_attention_2"
    except Exception as exc:  # noqa: BLE001
        marker("model_load", "fa2_unavailable", err=str(exc)[:200])
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
            device_map="cuda", trust_remote_code=False, attn_implementation="sdpa")
        RECEIPT["attn_implementation"] = "sdpa"
    model.config.use_cache = False
    model.eval()
    RECEIPT["model_load_s"] = round(time.time() - t0, 2)
    flush_receipt()
    marker("model_load", "done", seconds=RECEIPT["model_load_s"], attn=RECEIPT["attn_implementation"])

    assert_shared_render_invariant(tok, system)

    for name, param in model.named_parameters():
        if any(part in name.lower() for part in ("vision", "visual", "image_processor", "merger")):
            param.requires_grad = False
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules="all-linear",
                      lora_dropout=0.05, trust_remote_code=False)
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    RECEIPT["params"] = {"total": total, "trainable": trainable,
                         "trainable_pct": round(trainable / total * 100, 4)}
    flush_receipt()
    marker("lora", "done", trainable=trainable, pct=RECEIPT["params"]["trainable_pct"])

    ds = Dataset.from_list([
        {"messages": r["messages"], "chat_template_kwargs": {"enable_thinking": False}}
        for r in rows])
    RECEIPT["data"] = {"rows": len(ds), "source": DATA, "source_sha256": sha256_file(DATA)}
    flush_receipt()
    marker("dataset", "done", rows=len(ds))

    cfg = SFTConfig(
        output_dir=str(OUT / "trainer"),
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=75,
        save_only_model=False,
        max_length=MAX_LEN,
        packing=False,
        dataset_text_field="text",
        assistant_only_loss=True,
        report_to=[],
        seed=20260828,
        data_seed=20260828,
    )
    t1 = time.time()
    marker("train", "start", steps=MAX_STEPS)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    history = [{"step": h.get("step"), "loss": h.get("loss"), "grad_norm": h.get("grad_norm"),
                "learning_rate": h.get("learning_rate")} for h in trainer.state.log_history if "loss" in h]
    RECEIPT["train"] = {"seconds": round(time.time() - t1, 1), "max_steps": MAX_STEPS,
                        "loss_history_tail": history[-8:], "log_history_len": len(history)}
    flush_receipt()
    marker("train", "done", seconds=RECEIPT["train"]["seconds"],
           last_loss=history[-1]["loss"] if history else None)

    marker("merge", "start")
    merged_model = model.merge_and_unload()
    merged_dir = OUT / "merged"
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    del merged_model, model, trainer
    torch.cuda.empty_cache()
    marker("merge", "done", dir=str(merged_dir))

    t2 = time.time()
    marker("reload", "start")
    eval_model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False)
    eval_model.eval()
    RECEIPT["reload_s"] = round(time.time() - t2, 2)
    flush_receipt()
    marker("reload", "done", seconds=RECEIPT["reload_s"])

    questions = load_questions(QUESTIONS, EVAL_LIMIT)
    eval_model.eval()
    tok.padding_side = "left"
    prompts = [render_context_prefix(tok, system, q["prompt"]) for q in questions]
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(prompts), EVAL_BATCH):
            batch = prompts[start:start + EVAL_BATCH]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                         max_length=MAX_LEN).to(eval_model.device)
            width = inputs["input_ids"].shape[1]
            tb = time.time()
            seqs = eval_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
            elapsed = time.time() - tb
            for q, seq in zip(questions[start:start + EVAL_BATCH], seqs):
                text = tok.decode(seq[width:], skip_special_tokens=False)
                parsed = strict_parse(text)
                results.append({"problem_id": q["id"], "tokens": int(seq[width:].shape[0]),
                                "seconds": round(elapsed / len(batch), 2), "parse": parsed, "raw": text})
            marker("eval_batch", "done", done=len(results), seconds=round(elapsed, 1))
    successes = [x for x in results if x["parse"]["status"] == "success"]
    failures = [x for x in results if x["parse"]["status"] != "success"]
    RECEIPT["eval"] = {
        "questions": len(results), "success": len(successes), "failure_or_uncertain": len(failures),
        "align_prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
        "prompt_tokens": len(tok(prompts[0], add_special_tokens=False)["input_ids"]),
        "mean_tokens": round(sum(x["tokens"] for x in results) / max(1, len(results)), 1),
        "reason_counts": {r: sum(1 for x in failures if x["parse"]["reason"] == r)
                          for r in sorted({x["parse"]["reason"] for x in failures})},
    }
    (OUT / "success_raw.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in successes[:5]), encoding="utf-8")
    (OUT / "failure_raw.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in failures[:5]), encoding="utf-8")
    flush_receipt()
    summary = {"eval_success": len(successes), "eval_total": len(results),
               "mean_tokens": RECEIPT["eval"]["mean_tokens"],
               "reason_counts": RECEIPT["eval"]["reason_counts"]}
    print("**R3-3_SUMMARY** " + json.dumps(summary, ensure_ascii=False), flush=True)
    marker("receipt", "done", path=str(OUT / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
