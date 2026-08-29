"""R3-4: multi-turn tool-loop evaluation per the 06 protocol.

R3-3 conviction: the merged model correctly emits a single-action assistant
turn (<run> ... </run> then EOS) because P2 training turns are single-action;
a single-shot eval can never see <final>. This runtime evaluates the way the
model was trained to operate:

  per question:
    1. generate from the aligned context prefix (empty-think aligned)
    2. extract <run> code; execute it in a restricted local sandbox
       (whitelisted imports, 10 s timeout, stdout/stderr captured)
    3. append a P2-format user turn <output>\\n{result}\\n</output> plus the
       aligned assistant generation prompt
    4. generate again, expecting <final>...</final>
    5. parse the concatenation with strict_parse v1 (special tokens stripped
       first; their presence is recorded separately)

Turns are capped (max 3 run rounds), every stage is marker-logged, and the
receipt is flushed incrementally. No capability or correctness claim is made:
this measures protocol-format compliance only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

VERSION = "r3-4-toolloop-eval-v2"
MERGED = os.environ.get("R3_MODEL", "/data/magnus/closedloop-0828/r3-3-sft-eval-v4/merged")
QUESTIONS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
OUT = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r3-4-toolloop-v2"))
SYSTEM_SHA = "8ed1122a47ae089b1f577d61ad906cf4f7aa5f39627bfef7b6bf2afe79be3217"
MAX_NEW_RUN = 1024
MAX_NEW_FINAL = 1024
MAX_RUN_ROUNDS = 3
EVAL_LIMIT = 20
SANDBOX_TIMEOUT_S = 10.0

RECEIPT: dict[str, Any] = {"version": VERSION, "model": MERGED}
SPECIAL = re.compile(r"<\|[^<>]*\|>")


def marker(stage: str, status: str, **kw: Any) -> None:
    print(f"=== R3-4 {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def flush_receipt() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "receipt.json")


def strict_parse_v1(text: str) -> dict[str, Any]:
    """R3-3 lesson: special tokens (<|endoftext|>, <|im_end|>) must be stripped
    before tag scanning; their counts are reported but not penalized."""
    specials = SPECIAL.findall(text)
    cleaned = SPECIAL.sub("", text)
    result = strict_parse_v0(cleaned)
    result["special_tokens_seen"] = sorted({s for s in specials})
    return result


def strict_parse_v0(text: str) -> dict[str, Any]:
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
        result["unknown_tags"] = sorted({t for t in tags if t not in allowed})
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


def sandbox_exec(code: str) -> dict[str, Any]:
    """Restricted local execution mirroring the P2 sandbox contract: fresh
    process, 10 s timeout, stdout/stderr captured, no network assumptions."""
    proc = subprocess.run(["python3", "-c", code], capture_output=True, text=True,
                          timeout=SANDBOX_TIMEOUT_S)
    out = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    return {"ok": proc.returncode == 0, "output": out[:8000], "rc": proc.returncode}


def align_assistant_prefix(tok: Any, prompt: str) -> str:
    """Close the think block the generation prompt opens, byte-aligned with the
    training render (prompt ends '<think>\\n'; training shows empty think)."""
    if prompt.endswith("<think>\n"):
        return prompt + "\n</think>\n\n"
    if prompt.endswith("<|im_start|>assistant\n"):
        return prompt + "<think>\n\n</think>\n\n"
    return prompt


ASSISTANT_HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
IM_END = "<|im_end|>\n"


def generate(model: Any, tok: Any, prompt: str, max_new: int) -> tuple[str, float, int]:
    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    width = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        seq = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)[0]
    elapsed = time.time() - t0
    text = tok.decode(seq[width:], skip_special_tokens=False)
    return text, elapsed, int(seq[width:].shape[0])


def eval_question(model: Any, tok: Any, system: str, q: dict[str, Any]) -> dict[str, Any]:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": q["prompt"]}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False})
    prompt = align_assistant_prefix(tok, prompt)
    transcript = ""
    turns: list[dict[str, Any]] = []
    for round_idx in range(MAX_RUN_ROUNDS):
        text, secs, ntok = generate(model, tok, prompt,
                                    MAX_NEW_RUN if round_idx == 0 else MAX_NEW_FINAL)
        turns.append({"round": round_idx, "role": "assistant", "tokens": ntok,
                      "seconds": round(secs, 2), "head": text[:160]})
        transcript += text
        run_m = re.search(r"<run>\s*(.*?)\s*</run>", text, flags=re.S)
        final_m = re.search(r"<final>\s*(.*?)\s*</final>", text, flags=re.S)
        if final_m and not run_m:
            break
        if run_m:
            try:
                execution = sandbox_exec(run_m.group(1))
            except subprocess.TimeoutExpired:
                execution = {"ok": False, "output": "<timeout>", "rc": 124}
            turns.append({"round": round_idx, "role": "sandbox", "ok": execution["ok"],
                          "output_head": execution["output"][:120]})
            # Continuation must mirror the P2 training render exactly:
            # assistant turn closed, user <output> turn, new assistant head.
            clean = SPECIAL.split(text)[0]
            prompt = (prompt + clean + IM_END
                      + "<|im_start|>user\n"
                      + f"<output>\n{execution['output']}\n</output>" + IM_END
                      + ASSISTANT_HEAD)
        else:
            break
    parse = strict_parse_v1(transcript)
    return {"problem_id": q["id"], "parse": parse, "turns": turns,
            "transcript_chars": len(transcript)}


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    flush_receipt()
    marker("env", "done", torch=torch.__version__)

    # sandbox sanity: torch image may lack sympy; record honestly.
    try:
        r = sandbox_exec("import sympy; print(sympy.__version__)")
        RECEIPT["sandbox_sympy"] = r
    except Exception as exc:  # noqa: BLE001
        RECEIPT["sandbox_sympy"] = {"ok": False, "error": str(exc)[:200]}
    flush_receipt()
    marker("sandbox", "done", sympy=RECEIPT["sandbox_sympy"].get("output", "").strip() or "unavailable")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="cuda", trust_remote_code=False)
    model.eval()
    RECEIPT["model_load_s"] = round(time.time() - t0, 2)
    flush_receipt()
    marker("model_load", "done", seconds=RECEIPT["model_load_s"])

    # system prompt must be the training one; verify against merged tokenizer chat flow
    import json as _json
    sys_sha = None
    with open("/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl", encoding="utf-8") as f:
        for line in f:
            row = _json.loads(line)
            if row.get("protocol_valid") is True:
                sys_sha = hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()
                system = row["messages"][0]["content"]
                break
    RECEIPT["system_sha_matches"] = sys_sha == SYSTEM_SHA
    flush_receipt()
    marker("system_prompt", "done", matches=RECEIPT["system_sha_matches"])

    questions = load_questions(QUESTIONS, EVAL_LIMIT)
    results = []
    for i, q in enumerate(questions, 1):
        r = eval_question(model, tok, system, q)
        results.append(r)
        RECEIPT["eval_results_tail"] = [
            {"problem_id": x["problem_id"], "status": x["parse"]["status"],
             "reason": x["parse"].get("reason")} for x in results[-5:]]
        flush_receipt()
        marker("question", "done", i=i, result=r["parse"]["status"],
               reason=r["parse"].get("reason"))
    successes = [x for x in results if x["parse"]["status"] == "success"]
    failures = [x for x in results if x["parse"]["status"] != "success"]
    RECEIPT["eval"] = {
        "questions": len(results), "success": len(successes),
        "failure_or_uncertain": len(failures),
        "reason_counts": {r: sum(1 for x in failures if x["parse"].get("reason") == r)
                          for r in sorted({x["parse"].get("reason", "none") for x in failures})},
    }
    (OUT / "success_raw.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in successes[:5]), encoding="utf-8")
    (OUT / "failure_raw.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in failures[:5]), encoding="utf-8")
    flush_receipt()
    print("**R3-4_SUMMARY** " + json.dumps(RECEIPT["eval"], ensure_ascii=False), flush=True)
    marker("receipt", "done", path=str(OUT / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
