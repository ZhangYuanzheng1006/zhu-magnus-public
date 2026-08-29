"""R3-2: teacher 27B Transformers generation entry with explicit stage markers.

Uses the load parameters validated on the 9B probe (dtype=, low_cpu_mem_usage,
use_safetensors) plus a bounded 8192-token context, generates 512 tokens for a
single prompt, and flushes a partial receipt after every stage so a mid-run
crash still leaves evidence on /data. Stage markers are printed unbuffered.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

VERSION = "r3-2-teacher-gen-v1"
MODEL = "/data/magnus/models/Qwen3.8-27B-20260828"
QUESTIONS = "/data/magnus/closedloop-0828/p1/problems.jsonl"
OUT_DIR = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r3-2-teacher-gen-v1"))
MAX_PROMPT_TOKENS = 8192
MAX_NEW_TOKENS = 512

RECEIPT: dict[str, Any] = {"version": VERSION, "model": MODEL, "stages": {}}


def marker(stage: str, status: str, **kw: Any) -> None:
    print(f"=== R3-2 {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def flush_receipt() -> None:
    tmp = OUT_DIR / "receipt.json.tmp"
    tmp.write_text(json.dumps(RECEIPT, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_DIR / "receipt.json")


def load_question() -> dict[str, str]:
    with Path(QUESTIONS).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") in {"dev", "eval", "test", "holdout_family"}:
                return {"id": str(row.get("id", row.get("problem_id", "unknown"))),
                        "prompt": str(row.get("prompt", row.get("question", "")))}
    raise RuntimeError("no eligible question found")


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT["torch"] = torch.__version__
    RECEIPT["cuda"] = torch.version.cuda
    RECEIPT["gpu"] = torch.cuda.get_device_name(0)
    flush_receipt()
    marker("env", "done", torch=torch.__version__, cuda=torch.version.cuda)

    question = load_question()
    RECEIPT["question"] = {"id": question["id"], "prompt_sha256":
                           hashlib.sha256(question["prompt"].encode()).hexdigest()}
    flush_receipt()
    marker("question", "done", qid=question["id"])

    t0 = time.time()
    marker("tok_load", "start")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    marker("tok_load", "done", seconds=round(time.time() - t0, 2))

    t1 = time.time()
    marker("model_load", "start", params="dtype=bfloat16,low_cpu_mem_usage=True,use_safetensors=True,device_map=cuda")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cuda",
        trust_remote_code=False,
    )
    model.eval()
    load_s = round(time.time() - t1, 2)
    n_params = sum(p.numel() for p in model.parameters())
    RECEIPT["stages"]["model_load"] = {"seconds": load_s, "params": n_params,
                                       "mem_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}
    flush_receipt()
    marker("model_load", "done", seconds=load_s, params=n_params,
           mem_gb=RECEIPT["stages"]["model_load"]["mem_gb"])

    messages = [{"role": "user", "content": question["prompt"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   chat_template_kwargs={"enable_thinking": True,
                                                         "reasoning_effort": "medium"})
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).to(model.device)
    RECEIPT["stages"]["prompt"] = {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                                   "max_prompt_tokens": MAX_PROMPT_TOKENS}
    flush_receipt()
    marker("prompt", "done", prompt_tokens=int(inputs["input_ids"].shape[1]))

    t2 = time.time()
    marker("generate", "start", max_new_tokens=MAX_NEW_TOKENS)
    with torch.no_grad():
        seqs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.pad_token_id)
    gen_s = round(time.time() - t2, 2)
    width = int(inputs["input_ids"].shape[1])
    seq = seqs[0]
    n_new = int(seq[width:].shape[0])
    raw = tok.decode(seq[width:], skip_special_tokens=False)
    RECEIPT["stages"]["generate"] = {"seconds": gen_s, "new_tokens": n_new,
                                     "tokens_per_second": round(n_new / gen_s, 3) if gen_s else None}
    RECEIPT["raw"] = raw
    RECEIPT["raw_chars"] = len(raw)
    flush_receipt()
    marker("generate", "done", seconds=gen_s, new_tokens=n_new,
           tokens_per_second=RECEIPT["stages"]["generate"]["tokens_per_second"])

    print("=== R3-2 GEN_TEXT begin ===", flush=True)
    print(raw, flush=True)
    print("=== R3-2 GEN_TEXT end ===", flush=True)
    marker("receipt", "done", path=str(OUT_DIR / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
