"""Hard startup gates for the Qwen3.5 formal training runs."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

REQUIRED = {
    "transformers": "5.16.1",
    "trl": "1.12.0",
    "peft": "0.20.0",
    "flash-linear-attention": "0.4.1",
    "causal-conv1d": "1.7.0",
    "flash-attn": "2.7.4.post1",
    "liger-kernel": "0.8.0",
}


def _version(name: str) -> str:
    return importlib.metadata.version(name)


def _find_vision_parameters(model) -> list[tuple[str, bool]]:
    return [
        (name, bool(param.requires_grad))
        for name, param in model.named_parameters()
        if any(part in name.lower() for part in ("vision", "visual", "image_processor", "merger"))
    ]


def parameter_census(model) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    return {
        "total": total,
        "trainable": sum(size for _, size in trainable),
        "trainable_pct": 100.0 * sum(size for _, size in trainable) / max(1, total),
        "trainable_names": [name for name, _ in trainable],
    }


def _callable_evidence(fn: Any) -> dict[str, Any]:
    """Record wrapper metadata without treating it as runtime dispatch proof."""
    cells = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
            cells.append({"type": type(value).__name__, "module": getattr(value, "__module__", None), "name": getattr(value, "__name__", None)})
        except ValueError:
            cells.append({"type": "empty"})
    return {"module": fn.__module__, "name": fn.__name__, "closure": cells}


def _kernel_imports() -> dict[str, Any]:
    result = {}
    for name in ("fla", "flash_attn", "causal_conv1d", "liger_kernel"):
        try:
            mod = importlib.import_module(name)
            result[name] = {"imported": True, "version": getattr(mod, "__version__", None), "module": mod.__name__}
        except Exception as exc:
            result[name] = {"imported": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def run_startup_gate(model=None, *, require_model: bool = False, attn_implementation: str | None = None) -> dict[str, Any]:
    """Fail closed if optimized kernels/pins/frozen vision conditions fail."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    result: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "versions": {},
        "kernel_imports": _kernel_imports(),
        "kernel_wrappers": {
            "chunk_gated_delta_rule": _callable_evidence(qwen.torch_chunk_gated_delta_rule),
            "causal_conv1d": _callable_evidence(qwen.causal_conv1d_fn),
        },
        "dispatch_evidence": "wrapper/import metadata only; actual model forward required",
        "parameter_census": None,
        "vision_parameters": [],
        "flash_attention_2_requested": attn_implementation == "flash_attention_2",
        "passed": False,
        "errors": [],
    }
    if not result["cuda_available"]:
        result["errors"].append("CUDA is unavailable")
    if torch.version.cuda != "12.4":
        result["errors"].append(f"torch CUDA={torch.version.cuda}, expected 12.4")
    for package, expected in REQUIRED.items():
        try:
            actual = _version(package)
            result["versions"][package] = actual
            if package == "liger-kernel":
                parts = actual.split(".")[:2]
                ok = tuple(int(x) for x in parts) >= (0, 8)
            else:
                ok = actual == expected
            if not ok:
                result["errors"].append(f"{package}={actual}, expected {expected} or newer")
        except Exception as exc:
            result["errors"].append(f"{package}: {type(exc).__name__}: {exc}")
    for name, info in result["kernel_imports"].items():
        if not info.get("imported"):
            result["errors"].append(f"{name} import failed: {info.get('error')}")
    if attn_implementation is not None and attn_implementation != "flash_attention_2":
        result["errors"].append(f"attention implementation={attn_implementation}, expected flash_attention_2")

    # The wrapper metadata is diagnostic, not proof of a compiled dispatch.
    # The training caller must record a real Qwen3.5 forward after this gate.
    if model is not None:
        result["parameter_census"] = parameter_census(model)
        result["vision_parameters"] = _find_vision_parameters(model)
        unfrozen = [name for name, trainable in result["vision_parameters"] if trainable]
        if unfrozen:
            result["errors"].append(f"vision tower not frozen: {unfrozen[:20]}")
        result["dispatch_evidence"] = "model loaded; forward/dispatch parity must be recorded by caller"
    elif require_model:
        result["errors"].append("model is required for parameter and vision freeze gates")

    result["passed"] = not result["errors"]
    return result


def write_gate(result: dict[str, Any], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_startup_gate(result: dict[str, Any]) -> None:
    if not result.get("passed"):
        raise RuntimeError("startup gate failed: " + "; ".join(result.get("errors", [])))
