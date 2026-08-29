"""R3-1: format 0/20 conviction — CPU autopsy of R2-2 failure samples.

Prints every saved R2-2 failure raw sample with structural annotations, dumps a
real P2 training row (the TRL conversational source), renders both training and
inference prompt paths with the merged checkpoint's own tokenizer, and compares
them token by token. Also fingerprints the installed TRL version's handling of
``messages`` / ``chat_template_kwargs``. Output goes to stdout and a receipt
under the output directory. Read-only with respect to all model weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VERSION = "r3-1-format-autopsy-v1"
OUT_DIR = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r3-1-format-autopsy-v1"))
R22_DIR = Path("/data/magnus/closedloop-0828/transformers-eval-v0/student-r2-2")
TRAJ = Path("/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl")
PROBLEMS = Path("/data/magnus/closedloop-0828/p1/problems.jsonl")
MERGED_TOK = Path("/data/magnus/models/Qwen3.5-9B-sft-20260828/merged")

# Verbatim from public/scripts/0828_transformers_eval.py (inference-side system).
EVAL_SYSTEM_PROMPT = (
    "你是数学推导助手。可用 <run>代码</run> 进行推导，最后必须以唯一的 "
    "<final>答案</final> 收束；run 每次全文重写，禁止 shell/网络。遇到异常、" 
    "超时、条件不足或结果矛盾时，必须在 final 中明确写无法确定。"
)

FINDINGS: dict[str, Any] = {"version": VERSION, "sections": {}}


def marker(section: str, status: str, **kw: Any) -> None:
    print(f"=== R3-1 {section} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ===", flush=True)


def classify(raw: str, tokens: int) -> list[str]:
    tags: list[str] = []
    if not raw.strip():
        tags.append("empty_output")
    if "<run>" in raw or "</run>" in raw:
        tags.append("has_run_tag")
    if "<output>" in raw:
        tags.append("has_output_tag")
    if "<final>" in raw or "</final>" in raw:
        tags.append("has_final_tag")
    if "<think>" in raw:
        tags.append("has_think_open")
    if "</think>" in raw:
        tags.append("has_think_close")
    if re.search(r"```", raw):
        tags.append("markdown_fence")
    if re.search(r"^#{1,3} ", raw, re.M):
        tags.append("markdown_heading")
    if "<|im_end|>" in raw:
        tags.append("ends_im_end")
    else:
        tags.append("no_eos_seen_maybe_truncated")
    if re.search(r"[\u4e00-\u9fff]", raw):
        tags.append("contains_cjk")
    letters = sum(ch.isalpha() for ch in raw)
    if raw and letters / max(1, len(raw)) < 0.15 and "contains_cjk" not in tags:
        tags.append("suspect_garbled_low_alpha_ratio")
    if tokens >= 4000:
        tags.append("near_max_budget")
    return tags


def autopsy_failures() -> None:
    sec: dict[str, Any] = {}
    sample_files = sorted(R22_DIR.glob("checkpoint-*/failure_raw.jsonl"))
    sec["files_found"] = [str(p) for p in sample_files]
    for path in sample_files:
        print(f"\n######## {path.parent.name} / {path.name} ########", flush=True)
        entries = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get("raw", "")
            tags = classify(raw, int(row.get("tokens", 0)))
            parse = row.get("parse", {})
            entries.append({"problem_id": row.get("problem_id"), "tokens": row.get("tokens"),
                            "seconds": row.get("seconds"), "parse": parse, "tags": tags})
            print(f"\n--- sample {i} problem_id={row.get('problem_id')} tokens={row.get('tokens')} "
                  f"parse={json.dumps(parse, ensure_ascii=False)} tags={tags}", flush=True)
            print("RAW-BEGIN (repr, control chars visible)", flush=True)
            print(repr(raw), flush=True)
            print("RAW-END", flush=True)
        sec[path.parent.name] = entries
    FINDINGS["sections"]["failures"] = sec
    marker("failures", "done", files=len(sample_files))


def dump_training_row() -> dict[str, Any] | None:
    if not TRAJ.exists():
        marker("training_row", "missing", path=str(TRAJ))
        return None
    row = None
    with TRAJ.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("protocol_valid") is True:
                row = r
                break
    if row is None:
        marker("training_row", "no_protocol_valid_row")
        return None
    print("\n######## training row (first protocol_valid P2 trajectory) ########", flush=True)
    print("row keys:", sorted(row.keys()), flush=True)
    print("has 'text' field:", "text" in row, flush=True)
    print("row system_prompt_sha256:", row.get("system_prompt_sha256"), flush=True)
    msgs = row.get("messages", [])
    train_system = msgs[0]["content"] if msgs and msgs[0].get("role") == "system" else None
    sha = hashlib.sha256(train_system.encode()).hexdigest() if train_system else None
    print("embedded system sha256:", sha, flush=True)
    print("system matches row system_prompt_sha256:", sha == row.get("system_prompt_sha256"), flush=True)
    if train_system is not None:
        print("--- TRAIN SYSTEM PROMPT FULL BEGIN ---", flush=True)
        print(train_system, flush=True)
        print("--- TRAIN SYSTEM PROMPT FULL END ---", flush=True)
    for i, m in enumerate(msgs[1:], 1):
        content = m.get("content", "")
        print(f"--- msg[{i}] role={m.get('role')} len={len(content)} head={content[:200]!r} "
              f"tail={content[-200:]!r}", flush=True)
    info = {"has_text_field": "text" in row,
            "row_system_sha": row.get("system_prompt_sha256"),
            "embedded_system_sha": sha, "n_messages": len(msgs)}
    FINDINGS["sections"]["training_row"] = info
    marker("training_row", "done")
    return row


def render_comparison(row: dict[str, Any] | None) -> None:
    if row is None:
        marker("render_compare", "skipped_no_training_row")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MERGED_TOK), trust_remote_code=False)
    msgs = row["messages"]

    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    # T1: what TRL's conversational path renders by default (no extra kwargs).
    t1 = tok.apply_chat_template(msgs, tokenize=False)
    # T2: with enable_thinking=False (what the training job INTENDED via the
    # chat_template_kwargs dataset column).
    try:
        t2 = tok.apply_chat_template(msgs, tokenize=False, chat_template_kwargs={"enable_thinking": False})
    except Exception as exc:  # noqa: BLE001
        t2 = None
        print("T2 render failed:", type(exc).__name__, exc, flush=True)
    # E: the R2-2 inference render (eval system prompt, generation prompt, no thinking).
    eval_msgs = [{"role": "system", "content": EVAL_SYSTEM_PROMPT}, msgs[1]]
    e = tok.apply_chat_template(eval_msgs, tokenize=False, add_generation_prompt=True,
                                chat_template_kwargs={"enable_thinking": False})

    print("\n######## render comparison ########", flush=True)
    print("T1 head:", repr(t1[:400]), flush=True)
    print("T1 tail:", repr(t1[-400:]), flush=True)
    print("T1 contains <think>:", "<think>" in t1, "| contains <run>:", "<run>" in t1,
          "| contains <final>:", "<final>" in t1, flush=True)
    if t2 is not None:
        print("T2 contains <think>:", "<think>" in t2, flush=True)
        print("T1 == T2:", t1 == t2, flush=True)
    print("E head:", repr(e[:400]), flush=True)
    print("E tail:", repr(e[-300:]), flush=True)

    i1, ie = ids(t1), ids(e)
    # Longest common prefix between training text and eval prompt.
    n = 0
    for a, b in zip(i1, ie):
        if a != b:
            break
        n += 1
    info = {
        "train_text_tokens": len(i1), "eval_prompt_tokens": len(ie),
        "common_prefix_tokens": n,
        "train_sha256": hashlib.sha256(t1.encode()).hexdigest(),
        "eval_prompt_sha256": hashlib.sha256(e.encode()).hexdigest(),
        "system_prompt_differs": msgs[0]["content"] != EVAL_SYSTEM_PROMPT,
        "t1_has_think": "<think>" in t1,
        "t2_equals_t1": (t2 == t1) if t2 is not None else None,
    }
    # Divergent region detail.
    div = i1[n:n + 8]
    info["train_tokens_after_prefix_first8"] = tok.convert_ids_to_tokens(div)
    print("common prefix tokens:", n, "/", len(i1), "train,", len(ie), "eval", flush=True)
    print("first train tokens after divergence:", info["train_tokens_after_prefix_first8"], flush=True)
    print("SYSTEM_PROMPT differs (train vs eval):", info["system_prompt_differs"], flush=True)
    print("train system head:", repr(msgs[0]["content"][:120]), flush=True)
    print("eval  system head:", repr(EVAL_SYSTEM_PROMPT[:120]), flush=True)
    FINDINGS["sections"]["render_compare"] = info
    marker("render_compare", "done", common_prefix=n)


def eval_question_prompt_check() -> None:
    """Confirm which eval system prompt the failing samples were rendered with."""
    if not PROBLEMS.exists():
        marker("question_check", "missing")
        return
    count = 0
    with PROBLEMS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") in {"dev", "eval", "test", "holdout_family"}:
                count += 1
                if count == 1:
                    print("first eval question head:", repr(str(row.get("prompt", row.get("question", "")))[:200]), flush=True)
    FINDINGS["sections"]["question_check"] = {"eligible_questions": count}
    marker("question_check", "done", eligible=count)


def trl_fingerprint() -> None:
    try:
        import trl
        info = {"trl_version": trl.__version__}
        src = Path(trl.__file__).parent / "trainer" / "sft_trainer.py"
        if src.exists():
            text = src.read_text(encoding="utf-8", errors="replace")
            hits = []
            for pat in ("chat_template_kwargs", "dataset_text_field", '"messages"', "'messages'"):
                lines = [ln.strip() for ln in text.splitlines() if pat in ln]
                hits.append({pat: lines[:6]})
            info["sft_trainer_hits"] = hits
        FINDINGS["sections"]["trl"] = info
        print("\n######## TRL fingerprint ########", flush=True)
        print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)
        marker("trl", "done")
    except Exception as exc:  # noqa: BLE001
        FINDINGS["sections"]["trl"] = {"error": f"{type(exc).__name__}: {exc}"}
        marker("trl", "fail", err=str(exc))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    autopsy_failures()
    row = dump_training_row()
    render_comparison(row)
    eval_question_prompt_check()
    trl_fingerprint()
    FINDINGS["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "receipt.json").write_text(json.dumps(FINDINGS, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**R3-1_SUMMARY** " + json.dumps({k: v for k, v in FINDINGS["sections"].items()
          if k in ("render_compare", "training_row", "question_check", "trl")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
