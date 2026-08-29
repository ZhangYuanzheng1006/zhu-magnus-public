"""F2: multi-turn 06-protocol rollout executor + minimal GRPO trainer.

Why this exists (05 review F2, R4-1 evidence): TRL GRPOTrainer generates
single-turn completions; on our multi-turn distribution that yielded 83%
no_final and zero_advantage_rate=100%. Production P4 rollouts must run the
full loop: assistant <run> -> real sandbox -> <output> user turn -> assistant
<final>.

Design (03b §3 config):
  - strict on-policy, updates_per_rollout=1 => PPO ratio is identically 1, so
    the GRPO objective reduces to a masked policy gradient with
    group-normalized advantage; we implement that directly (auditable, no
    framework surgery) and emit every O-observable from 03b §4.
  - Assistant-only credit: logprobs are taken on assistant-span token
    positions only; env (<output>) and template tokens are masked.
  - Reward: caller-supplied tier grader (03b §1 four tiers + abstain pair +
    uncertain mask).
  - O-08a (F4): custom clip-ratio callback emits custom/clip_ratio; with
    updates=1 it is identically 0 by construction and is emitted as such
    (recorded, not hidden).

This module is CPU-testable for bookkeeping; GPU paths need a checkpoint
(epoch-1 integration test, review F2 联调).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

SPECIAL = re.compile(r"<\|[^<>]*\|>")
IM_END = "<|im_end|>\n"
ASSISTANT_HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------------------------------------------------------- bookkeeping

@dataclass
class TurnSpan:
    role: str                # "assistant" | "env"
    text: str
    token_start: int         # [start, end) into the full sequence
    token_end: int


@dataclass
class RolloutResult:
    prompt: str
    token_ids: list[int]
    assistant_mask: list[bool]          # True where policy logprob is taken
    transcript: str
    turns: list[TurnSpan]
    tiers: dict[str, int] = field(default_factory=dict)
    reward: float = 0.0
    truncated: bool = False
    rounds: int = 0
    exec_seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def align_assistant_prefix(prompt: str) -> str:
    if prompt.endswith("<think>\n"):
        return prompt + "\n</think>\n\n"
    if prompt.endswith("<|im_start|>assistant\n"):
        return prompt + ASSISTANT_HEAD
    return prompt


def rollout_one(
    model: Any,
    tok: Any,
    system: str,
    question: str,
    sandbox_fn: Callable[[str], dict[str, Any]],
    grader: Callable[[str], tuple[float, dict[str, int]]],
    *,
    max_rounds: int = 3,
    max_new_run: int = 1024,
    max_new_final: int = 2048,
    episode_token_budget: int = 4096,
) -> RolloutResult:
    """One full 06-protocol episode. `grader(transcript)` returns
    (reward, tier_counter); reward side effects (verifier timings) stay in the
    grader. Token bookkeeping is exact: assistant spans are the only unmasked
    positions."""
    import torch

    base = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False})
    seq = tok(align_assistant_prefix(base), add_special_tokens=False)["input_ids"]
    transcript = ""
    turns: list[TurnSpan] = []
    mask: list[bool] = [False] * len(seq)
    tiers_total: dict[str, int] = {}
    reward = 0.0
    truncated = False
    exec_seconds = 0.0
    spent = 0

    for round_idx in range(max_rounds):
        budget = min(max_new_run if round_idx == 0 else max_new_final,
                     episode_token_budget - spent)
        if budget <= 0:
            truncated = True
            break
        ids = torch.tensor([seq], device=model.device)
        width = ids.shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=budget, do_sample=True,
                                 temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)[0]
        exec_seconds += time.time() - t0
        new_ids = out[width:].tolist()
        text = tok.decode(new_ids, skip_special_tokens=False)
        spent += len(new_ids)
        turns.append(TurnSpan("assistant", text, len(seq), len(seq) + len(new_ids)))
        mask.extend([True] * len(new_ids))
        seq.extend(new_ids)
        transcript += text
        run_m = re.search(r"<run>\s*(.*?)\s*</run>", text, flags=re.S)
        final_m = re.search(r"<final>\s*(.*?)\s*</final>", text, flags=re.S)
        if final_m and not run_m:
            break
        if run_m:
            ex = sandbox_fn(run_m.group(1))
            exec_seconds += ex.get("seconds", 0.0)
            output = ex["output"][:8000]
            env_text = f"<|im_start|>user\n<output>\n{output}\n</output><|im_end|>\n"
            env_ids = tok(env_text, add_special_tokens=False)["input_ids"]
            turns.append(TurnSpan("env", env_text, len(seq), len(seq) + len(env_ids)))
            mask.extend([False] * len(env_ids))
            seq.extend(env_ids)
            transcript += env_text
            head_ids = tok(ASSISTANT_HEAD, add_special_tokens=False)["input_ids"]
            seq.extend(head_ids)
            mask.extend([False] * len(head_ids))
            rounds = round_idx + 1
            if spent >= episode_token_budget:
                truncated = True
                break
            continue
        break
    else:
        truncated = True
    rounds = len([t for t in turns if t.role == "assistant"])

    reward, tiers = grader(transcript)
    tiers_total = tiers
    return RolloutResult(prompt=align_assistant_prefix(base), token_ids=seq,
                         assistant_mask=mask, transcript=transcript, turns=turns,
                         tiers=tiers_total, reward=reward, truncated=truncated,
                         rounds=rounds, exec_seconds=round(exec_seconds, 3),
                         meta={"spent_tokens": spent})


# ---------------------------------------------------------------- GRPO core

def group_advantages(rewards: list[float]) -> tuple[list[float], bool]:
    """Group-normalize; returns (advantages, is_zero_advantage_group)."""
    if not rewards:
        return [], True
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < 1e-6:
        return [0.0] * len(rewards), True
    return [(r - mean) / std for r in rewards], False


def masked_logprobs(model: Any, tok: Any, token_ids: list[int],
                    assistant_mask: list[bool], device: str = "cuda") -> list[float]:
    """Per-position logprob of the *next* token at assistant positions.
    Positions where mask[i-1] is False (prompt/env) contribute nothing."""
    import torch
    ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids[:, :-1]).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out: list[float] = []
    for pos in range(1, len(token_ids)):
        if assistant_mask[pos]:
            out.append(float(logprobs[pos - 1, token_ids[pos]]))
        else:
            out.append(0.0)
    out.append(0.0)
    return out


def grpo_step(model: Any, ref_model: Any, tok: Any, batch: list[RolloutResult],
              advantages: list[float], beta: float, optimizer: Any,
              device: str = "cuda") -> dict[str, Any]:
    """One strict-on-policy GRPO update: loss = -(A * mean_logprob) over
    assistant tokens, plus beta * token-level KL(p||ref) on the same mask.
    Ratio is identically 1 (updates=1); clip_ratio emitted as 0.0 (F4)."""
    import torch

    total_loss = 0.0
    kl_sum = 0.0
    n_tokens = 0
    optimizer.zero_grad()
    for res, adv in zip(batch, advantages):
        if adv == 0.0:
            continue
        ids = torch.tensor([res.token_ids], device=device)
        mask = torch.tensor(res.assistant_mask[1:] + [False], device=device)
        logits = model(input_ids=ids[:, :-1]).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(1, torch.tensor(res.token_ids[1:], device=device)
                               .unsqueeze(1)).squeeze(1)
        sel = tok_logp[mask]
        if sel.numel() == 0:
            continue
        pg = -(adv * sel.mean())
        kl_term = torch.tensor(0.0, device=device)
        if ref_model is not None and beta > 0:
            with torch.no_grad():
                ref_logits = ref_model(input_ids=ids[:, :-1]).logits[0]
            ref_logp = torch.log_softmax(ref_logits.float(), dim=-1)
            ref_sel = ref_logp.gather(1, torch.tensor(res.token_ids[1:], device=device)
                                      .unsqueeze(1)).squeeze(1)[mask]
            # k3 estimator (per-token p/ref - log(ref/p) - 1), low variance
            ratio = (sel - ref_sel).detach().exp()
            kl_term = (ratio - torch.log(ratio.clamp_min(1e-8)) - 1).mean()
        loss = pg + beta * kl_term
        loss.backward()
        total_loss += float(loss.detach())
        kl_sum += float(kl_term.detach())
        n_tokens += int(sel.numel())
    if n_tokens:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad()
    return {"loss": total_loss, "kl_mean": kl_sum / max(1, len(batch)),
            "policy_tokens": n_tokens,
            "custom/clip_ratio": 0.0}  # F4: identically 0 under updates=1
