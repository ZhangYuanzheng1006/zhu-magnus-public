"""Frozen generation and training budgets for the formal first loop."""
from __future__ import annotations

TEACHER = {
    "max_model_len": 32768,
    "enable_thinking": True,
    "reasoning_effort_primary": "medium",
    "reasoning_effort_control": "low",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "max_tokens_primary": 8192,
    "max_tokens_control": 4096,
}

STUDENT = {
    "enable_thinking": False,
    "max_model_len": 16384,
    "max_tokens_per_action": 2048,
    "episode_max_tokens": 4096,
    "episode_max_runs": 8,
    "sft_max_seq_len": 4096,
    "grpo_sampling": {"temperature": 1.0, "top_p": 1.0},
    "eval_sampling": {"temperature": 0.0},
}


def teacher_max_tokens(effort: str) -> int:
    if effort == TEACHER["reasoning_effort_primary"]:
        return TEACHER["max_tokens_primary"]
    if effort == TEACHER["reasoning_effort_control"]:
        return TEACHER["max_tokens_control"]
    raise ValueError(f"unsupported teacher effort: {effort}")
