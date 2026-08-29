"""Transformers-only evaluation for merged student checkpoints and teacher fallback.

This is deliberately a bounded observation harness, not a semantic correctness
verifier.  It uses the tokenizer's chat template and batched ``generate`` path.
Every artifact is written below a versioned /data directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

VERSION = "transformers-eval-v0"
DEFAULT_QUESTIONS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
DEFAULT_STUDENTS = [
    "/data/magnus/models/Qwen3.5-9B-sft-20260828/merged",
    "/data/magnus/models/Qwen3.5-9B-sft-20260828-accel/merged",
]
DEFAULT_OUT = "/data/magnus/closedloop-0828/transformers-eval-v0"
TEACHER_MODEL = "/data/magnus/models/Qwen3.8-27B-20260828"
SYSTEM_PROMPT = (
    "你是数学推导助手。可用 <run>代码</run> 进行推导，最后必须以唯一的 "
    "<final>答案</final> 收束；run 每次全文重写，禁止 shell/网络。遇到异常、" 
    "超时、条件不足或结果矛盾时，必须在 final 中明确写无法确定。"
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("student_r2_2", "teacher_fallback"), required=True)
    p.add_argument("--questions", default=os.environ.get("EVAL_QUESTIONS", DEFAULT_QUESTIONS))
    p.add_argument("--checkpoints", nargs="*", default=DEFAULT_STUDENTS)
    p.add_argument("--model", default=os.environ.get("TEACHER_MODEL", TEACHER_MODEL))
    p.add_argument("--out", default=os.environ.get("EVAL_OUT", DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    return p.parse_args()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_questions(path: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # Training rows are never silently reused as evaluation questions.
            if row.get("split") not in {"dev", "eval", "test", "holdout_family"}:
                continue
            row["id"] = str(row.get("id", row.get("problem_id", len(rows))))
            row["prompt"] = str(row.get("prompt", row.get("question", "")))
            if row["prompt"]:
                rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"no dev/eval/test questions found (training rows refused): {path}")
    return rows


def new_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Document the no-training-row invariant for the R2-2 input."""
    return [r for r in rows if r.get("split") in {"dev", "eval", "test", "holdout_family"}]


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


def messages(row: dict[str, Any], system: str | None = None) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        return row["messages"][:2]
    out = []
    if system:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": row["prompt"]})
    return out


def render(tok: Any, rows: list[dict[str, Any]], *, effort: str | None = None, system: str | None = None) -> list[str]:
    rendered = []
    for row in rows:
        kw: dict[str, Any] = {"enable_thinking": effort is not None}
        if effort:
            kw["reasoning_effort"] = effort
        rendered.append(tok.apply_chat_template(messages(row, system), tokenize=False,
                         add_generation_prompt=True, chat_template_kwargs=kw))
    return rendered


def generate_batches(model: Any, tok: Any, texts: list[str], max_new: int, batch_size: int, *, sample: bool = False) -> list[dict[str, Any]]:
    import torch
    outputs: list[dict[str, Any]] = []
    tok.padding_side = "left"
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                         max_length=getattr(tok, "model_max_length", 32768)).to(model.device)
            width = inputs["input_ids"].shape[1]
            t0 = time.perf_counter()
            try:
                kwargs = {"max_new_tokens": max_new, "do_sample": sample, "pad_token_id": tok.pad_token_id}
                if sample:
                    kwargs.update(temperature=0.7, top_p=0.95)
                seqs = model.generate(**inputs, **kwargs)
                elapsed = time.perf_counter() - t0
                for seq in seqs:
                    text = tok.decode(seq[width:], skip_special_tokens=False)
                    outputs.append({"text": text, "seconds": elapsed / max(1, len(seqs)),
                                    "tokens": int(seq[width:].shape[0]), "uncertain": False})
            except Exception as exc:
                outputs.extend({"text": "", "seconds": time.perf_counter() - t0,
                                "tokens": 0, "uncertain": True,
                                "error": f"{type(exc).__name__}: {exc}"} for _ in batch)
    return outputs


def load_model(path: str, *, teacher: bool = False) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=False)
    model.eval()
    return model, tok


def student_run(rows: list[dict[str, Any]], checkpoints: list[str], out: Path, batch: int, max_new: int, generations: int, question_source: str) -> dict[str, Any]:
    receipt: dict[str, Any] = {"version": VERSION, "mode": "student_r2_2", "semantic_parser": {"version": "v0", "limitation": "strict format only; no CAS, numeric, gold, or equivalence judgment"}, "checkpoints": {}}
    for ckpt_index, ckpt in enumerate(checkpoints, 1):
        model_out = out / f"checkpoint-{ckpt_index:02d}-{Path(ckpt).parent.name}"
        model_out.mkdir(parents=True, exist_ok=True)
        try:
            model, tok = load_model(ckpt)
            system = SYSTEM_PROMPT
            all_results = []
            for rep in range(generations):
                generated = generate_batches(model, tok, render(tok, rows, system=system), max_new, batch)
                for row, gen in zip(rows, generated):
                    parsed = {"status": "uncertain", "reason": "generation_exception"} if gen.get("uncertain") else strict_parse(gen["text"])
                    all_results.append({"problem_id": row["id"], "rep": rep, "parse": parsed,
                        "tokens": gen["tokens"], "seconds": round(gen["seconds"], 4), "raw": gen["text"]})
            successes = [x for x in all_results if x["parse"]["status"] == "success"]
            failures = [x for x in all_results if x["parse"]["status"] != "success"]
            (model_out / "success_raw.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in successes[:5]), encoding="utf-8")
            (model_out / "failure_raw.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in failures[:5]), encoding="utf-8")
            receipt["checkpoints"][ckpt] = {"count": len(all_results), "success": len(successes), "failure_or_uncertain": len(failures), "raw_samples": {"success": min(5, len(successes)), "failure": min(5, len(failures))}}
            del model
        except Exception as exc:
            receipt["checkpoints"][ckpt] = {"status": "uncertain", "error": f"{type(exc).__name__}: {exc}"}
    receipt["questions"] = {"count": len(rows), "source_sha256": sha256_file(question_source)}
    return receipt


def teacher_run(rows: list[dict[str, Any]], model_path: str, out: Path, batch: int, max_new: int) -> dict[str, Any]:
    model, tok = load_model(model_path, teacher=True)
    result: dict[str, Any] = {"version": VERSION, "mode": "teacher_fallback", "model": model_path, "semantic_parser": {"version": "v0", "limitation": "format/uncertainty observation only; no semantic correctness claim"}, "efforts": {}}
    for effort, budget in (("medium", max_new), ("low", min(max_new, 4096))):
        entries = []
        for row, gen in zip(rows[:3], generate_batches(model, tok, render(tok, rows[:3], effort=effort), budget, batch, sample=True)):
            text = gen["text"]
            think = re.search(r"<think>(.*?)</think>", text, re.S)
            think_text = think.group(1) if think else ""
            answer = text[think.end():] if think else text
            entries.append({"problem_id": row["id"], "effort": effort, "tokens": gen["tokens"], "seconds": round(gen["seconds"], 4), "tokens_per_second": round(gen["tokens"] / gen["seconds"], 3) if gen["seconds"] else None, "think": bool(think), "think_chars": len(think_text), "parse": strict_parse(answer), "raw": text})
        result["efforts"][effort] = entries
    out.mkdir(parents=True, exist_ok=True)
    (out / "teacher_fallback.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    a = args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = load_questions(a.questions, a.limit)
    receipt = student_run(rows, a.checkpoints, out, a.batch_size, a.max_new_tokens, a.generations, a.questions) if a.mode == "student_r2_2" else teacher_run(rows, a.model, out, a.batch_size, a.max_new_tokens)
    receipt["questions"] = len(rows)
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == "__main__":
    main()
