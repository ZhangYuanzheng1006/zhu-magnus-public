"""Compare two isolated Transformers model-loading keyword configurations.

Each configuration is executed in a fresh subprocess and loads the base model
independently. This is a load/API probe only; observations for this 9B model
must not be extrapolated to larger models.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL_PATH = "/data/magnus/models/Qwen3.5-9B-20260828"
DEFAULT_OUT = "/data/magnus/closedloop-0828/load-param-probe-9b"


def _record(config: str, model_path: str) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "config": config,
        "model_path": model_path,
        "scope": "Qwen3.5-9B base only; do not extrapolate to 27B",
        "status": "failed",
        "failure_stage": None,
        "error_type": None,
        "error": None,
        "tokenizer_load_wall_s": None,
        "model_load_wall_s": None,
        "first_forward_wall_s": None,
        "first_generate_wall_s": None,
        "parameter_count": None,
        "torch_version": None,
        "cuda_version": None,
        "cuda_device": None,
        "peak_gpu_memory_allocated_mb": None,
        "peak_gpu_memory_reserved_mb": None,
        "forward_evidence": None,
        "generate_evidence": None,
    }
    stage = "imports"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        result["cuda_device"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()

        stage = "tokenizer_load"
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        result["tokenizer_load_wall_s"] = round(time.perf_counter() - t0, 4)

        stage = "model_load"
        kwargs: dict[str, Any] = {
            "device_map": "cuda",
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if config == "A":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["dtype"] = torch.bfloat16
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        result["model_load_wall_s"] = round(time.perf_counter() - t0, 4)
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        model.eval()

        stage = "first_forward"
        prompt = "用一句话回答：1+1等于几？"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model(**inputs)
        result["first_forward_wall_s"] = round(time.perf_counter() - t0, 4)
        result["forward_evidence"] = {
            "logits_shape": list(output.logits.shape),
            "finite": bool(torch.isfinite(output.logits).all().item()),
        }

        stage = "first_generate"
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        result["first_generate_wall_s"] = round(time.perf_counter() - t0, 4)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        result["generate_evidence"] = {
            "token_count": int(generated.shape[-1]),
            "text_preview": decoded[:200],
            "nonempty": bool(decoded.strip()),
        }
        result["status"] = "success"
        result["failure_stage"] = None
    except Exception as exc:  # preserve the exact failed stage for diagnosis
        result["failure_stage"] = stage
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                result["peak_gpu_memory_allocated_mb"] = round(
                    torch.cuda.max_memory_allocated() / 2**20, 2
                )
                result["peak_gpu_memory_reserved_mb"] = round(
                    torch.cuda.max_memory_reserved() / 2**20, 2
                )
        except Exception:
            pass
    result["total_wall_s"] = round(time.perf_counter() - started, 4)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=("A", "B"), help="run one config in this process")
    p.add_argument("--model", default=os.environ.get("LOAD_PROBE_MODEL", MODEL_PATH))
    p.add_argument("--out", default=os.environ.get("LOAD_PROBE_OUT", DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.config:
        result = _record(args.config, args.model)
        (out / f"config_{args.config.lower()}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "success" else 1

    summary: list[dict[str, Any]] = []
    for config in ("A", "B"):
        config_out = out / f"config_{config.lower()}"
        config_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "--config", config, "--model", args.model, "--out", str(config_out)]
        completed = subprocess.run(cmd, check=False, text=True)
        result_path = config_out / f"config_{config.lower()}.json"
        if result_path.exists():
            summary.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            summary.append({"config": config, "status": "failed", "failure_stage": "child_process", "returncode": completed.returncode})
    payload = {"model_path": args.model, "configs": summary, "scope": "9B base probe only; no 27B extrapolation"}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if all(x.get("status") == "success" for x in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
