"""R2-0 runtime probe for one isolated vLLM candidate inside a Magnus job."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_DEFAULT = "/data/magnus/models/Qwen3.5-9B-20260828"
OUT_DEFAULT = "/data/magnus/closedloop-0828/r2-0-vllm-matrix"


def run(cmd: list[str], *, timeout: int) -> tuple[str, str, int | None]:
    try:
        proc = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, timeout=timeout,
        )
        return proc.stdout[-4000:], "completed", proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return output[-4000:], "timeout", None


def probe(interpreter: str, model: str, max_model_len: int, max_tokens: int) -> dict:
    code = r'''
import json, sys, time
result = {"import": "not_run", "load": "not_run", "generate": "not_run", "failure_stage": None}
try:
    import torch
    result.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  cuda_available=bool(torch.cuda.is_available()),
                  cuda_device_count=torch.cuda.device_count())
    if torch.cuda.is_available(): result["cuda_device_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    result["torch_probe_error"] = repr(exc)
try:
    import vllm
    result["vllm"] = vllm.__version__
    result["import"] = "pass"
except Exception as exc:
    result.update({"import": "fail", "failure_stage": "import", "error": repr(exc)})
    print(json.dumps(result)); raise SystemExit(0)
try:
    llm = vllm.LLM(model=sys.argv[1], trust_remote_code=False,
                   max_model_len=int(sys.argv[2]), tensor_parallel_size=1,
                   gpu_memory_utilization=0.90)
    result["load"] = "pass"
except Exception as exc:
    result.update({"load": "fail", "failure_stage": "load", "error": repr(exc)})
    print(json.dumps(result)); raise SystemExit(0)
try:
    params = vllm.SamplingParams(temperature=0.0, max_tokens=int(sys.argv[3]))
    t0 = time.time()
    output = llm.generate(["Give one concise mathematical identity and no explanation."], params)[0].outputs[0]
    result.update(generate="pass", generated_tokens=len(output.token_ids),
                  generate_seconds=round(time.time()-t0, 3), text_preview=output.text[:200])
except Exception as exc:
    result.update({"generate": "fail", "failure_stage": "generate", "error": repr(exc)})
print(json.dumps(result))
'''
    stdout, status, code_rc = run(
        [interpreter, "-c", code, model, str(max_model_len), str(max_tokens)], timeout=900
    )
    if status == "timeout":
        return {"import": "unknown", "load": "unknown", "generate": "unknown",
                "failure_stage": "timeout", "error": "candidate probe exceeded 900 seconds",
                "probe_log_tail": stdout}
    try:
        record = json.loads(stdout.strip().splitlines()[-1])
    except Exception:
        record = {"import": "unknown", "load": "unknown", "generate": "unknown",
                  "failure_stage": "probe_protocol", "probe_log_tail": stdout}
    if code_rc not in (0, None):
        record.setdefault("probe_returncode", code_rc)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--install-timeout", type=int, default=900)
    args = parser.parse_args()
    started = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    safe = args.candidate.replace("<", "lt").replace(">", "gt").replace("=", "").replace(",", "_")
    venv = out / ("venv-" + safe.replace(".", "_"))
    record = {"schema": "r2-0-vllm-matrix/v4", "candidate_spec": args.candidate,
              "model": args.model, "max_model_len": args.max_model_len,
              "requested_max_tokens": args.max_tokens, "venv": str(venv),
              "isolation": "candidate-specific venv with system site packages",
              "dependency_policy": "vllm/cloudpickle/pydantic/pydantic-core/typing-inspection --no-deps; preserve base-image torch",
              "pinned_python_deps": {"pydantic": "2.13.5", "pydantic-core": "2.46.5"},
              "protected_environment": "torch 2.5.1+cu124", "install": "not_run"}
    if not venv.exists():
        stdout, status, rc = run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], timeout=120)
        if status != "completed" or rc != 0:
            record.update(install="fail", failure_stage="venv", pip_log_tail=stdout)
            (out / (safe + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(record, ensure_ascii=False)); return 1
    py = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    stdout, status, rc = run([str(py), "-m", "pip", "install", "--no-cache-dir", "--no-deps",
                              args.candidate, "cloudpickle", "pydantic==2.13.5", "pydantic-core==2.46.5", "typing-inspection==0.4.2"], timeout=args.install_timeout)
    record["pip_log_tail"] = stdout
    if status != "completed" or rc != 0:
        record.update(install="timeout" if status == "timeout" else "fail", failure_stage="install")
    else:
        record["install"] = "pass"
        record.update(probe(str(py), args.model, args.max_model_len, args.max_tokens))
        if record.get("torch_cuda") != "12.4" or not str(record.get("torch", "")).startswith("2.5.1"):
            record["environment_warning"] = "base image torch/CUDA is not the required 2.5.1/cu124"
            record["failure_stage"] = "environment"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    (out / (safe + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record.get("failure_stage") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
