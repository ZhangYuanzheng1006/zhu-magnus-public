"""R3-0: torch 2.7.1 (cu126) base probe + vLLM 0.22.1 compatibility matrix.

Runs inside the verified cu124-train image. Builds an isolated venv
(--system-site-packages so image transformers/numpy stay visible while the
venv-local torch 2.7.1 shadows system torch 2.5.1), installs torch 2.7.1 from
a mirror chain, then walks the stage matrix:

  env -> venv -> torch_install -> cuda_matmul -> inductor_api
      -> vllm_install -> vllm_import -> vllm_load_generate

Every stage prints a ``=== R3-0 STAGE ... ===`` marker and is recorded in a
matrix dict that is written to the output directory and printed as a single
``**R3-0_MATRIX**`` JSON line. No stage is skipped; failures are recorded, not
raised. Outer success never implies vLLM compatibility.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

VERSION = "r3-0-torch27-probe-v1"
VENV_DIR = Path("/tmp/r3-0/venv")
WORK_DIR = Path("/tmp/r3-0")
OUT_DIR = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r3-0-torch27-v1"))
BASE_MODEL_CANDIDATES = [
    "/data/magnus/models/Qwen3.5-9B-20260828",
]
TORCH_VERSION = "2.7.1"
VLLM_VERSION = "0.22.1"
# Pure-python / pinned deps proven necessary by the R2-0 v2..v8 probe series.
VLLM_DEPS = [
    "cloudpickle",
    "pydantic==2.13.5",
    "pydantic-core==2.46.5",
    "typing-inspection==0.4.2",
    "annotated-types==0.7.0",
]
TORCH_ATTEMPTS = [
    ("pytorch_official_cu126", ["--index-url", "https://download.pytorch.org/whl/cu126"], 900),
    ("pypi_default", [], 900),
    ("aliyun_pytorch_wheels", ["-f", "https://mirrors.aliyun.com/pytorch-wheels/cu126/"], 900),
]

MATRIX: dict[str, Any] = {"version": VERSION, "torch": TORCH_VERSION, "vllm": VLLM_VERSION}


def marker(stage: str, status: str, **kw: Any) -> None:
    line = f"=== R3-0 STAGE {stage} {status} " + " ".join(f"{k}={v}" for k, v in kw.items()) + " ==="
    print(line, flush=True)


def record(stage: str, **kw: Any) -> None:
    MATRIX.setdefault("stages", {}).setdefault(stage, {}).update(kw)


def run(cmd: list[str], timeout: int, *, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return 124, out, str(exc)
    except Exception as exc:  # noqa: BLE001
        return 125, "", f"{type(exc).__name__}: {exc}"
    finally:
        print(f"--- cmd took {time.time() - t0:.1f}s: {' '.join(cmd[:6])} ...", flush=True)


def tail(text: str, limit: int = 3000) -> str:
    text = text or ""
    return text[-limit:]


def stage_env() -> None:
    t0 = time.time()
    rc, out, err = run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"], 60)
    info = out.strip().splitlines()
    record("env", driver_rows=info, seconds=round(time.time() - t0, 2), rc=rc)
    marker("env", "done" if info else "fail")


def stage_venv() -> bool:
    t0 = time.time()
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    rc, out, err = run(["python3", "-m", "venv", "--system-site-packages", str(VENV_DIR)], 300)
    ok = rc == 0 and (VENV_DIR / "bin" / "python").exists()
    record("venv", ok=ok, rc=rc, seconds=round(time.time() - t0, 2), err=tail(err, 800))
    marker("venv", "done" if ok else "fail")
    return ok


def torch_verify() -> tuple[bool, str]:
    code = (
        "import torch;"
        "info={'version':torch.__version__,'cuda':torch.version.cuda,"
        "'available':torch.cuda.is_available(),"
        "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None};"
        "import json;print(json.dumps(info))"
    )
    rc, out, err = run([str(VENV_DIR / "bin" / "python"), "-c", code], 120)
    ok = rc == 0 and '"available": true' in out.replace("'", '"')
    return ok, (out.strip().splitlines() or [""])[-1] + tail(err, 500)


def stage_torch_install() -> bool:
    for name, extra, timeout in TORCH_ATTEMPTS:
        t0 = time.time()
        marker("torch_install", "attempt", mirror=name)
        cmd = [str(VENV_DIR / "bin" / "pip"), "install", f"torch=={TORCH_VERSION}",
               "--timeout", "60", "--retries", "2", *extra]
        rc, out, err = run(cmd, timeout)
        if rc != 0:
            record("torch_install", **{f"rc_{name}": rc, f"err_{name}": tail(err, 800),
                                       f"seconds_{name}": round(time.time() - t0, 1)})
            marker("torch_install", "attempt_fail", mirror=name)
            continue
        ok, detail = torch_verify()
        record("torch_install", **{f"ok_{name}": ok, f"detail_{name}": detail[:1200],
                                   f"seconds_{name}": round(time.time() - t0, 1)})
        if ok:
            MATRIX["torch_source"] = name
            marker("torch_install", "done", mirror=name, detail=detail[:200])
            return True
        marker("torch_install", "verify_fail", mirror=name)
    MATRIX["torch_source"] = None
    marker("torch_install", "fail")
    return False


def stage_cuda_matmul() -> bool:
    t0 = time.time()
    code = (
        "import torch, time, json;"
        "a=torch.randn(2048,2048,device='cuda');b=torch.randn(2048,2048,device='cuda');"
        "torch.cuda.synchronize();t0=time.time();c=a@b;torch.cuda.synchronize();"
        "print(json.dumps({'ok':bool(torch.isfinite(c).all().item()),"
        "'matmul_s':round(time.time()-t0,4),'device':torch.cuda.get_device_name(0)}))"
    )
    rc, out, err = run([str(VENV_DIR / "bin" / "python"), "-c", code], 180)
    ok = rc == 0 and '"ok": true' in out.replace("'", '"')
    record("cuda_matmul", ok=ok, rc=rc, out=tail(out, 500), err=tail(err, 800),
           seconds=round(time.time() - t0, 2))
    marker("cuda_matmul", "done" if ok else "fail")
    return ok


def stage_inductor_api() -> None:
    code = "from torch._inductor import custom_graph_pass; print('custom_graph_pass OK')"
    rc, out, err = run([str(VENV_DIR / "bin" / "python"), "-c", code], 120)
    record("inductor_api", torch27=rc == 0, out=tail(out, 300), err=tail(err, 500))
    rc25, out25, err25 = run(["python3", "-c", code], 120)
    record("inductor_api", system_torch25=rc25 == 0, system_err=tail(err25, 300))
    marker("inductor_api", "done", torch27=rc == 0, torch25=rc25 == 0)


def stage_vllm_install() -> bool:
    t0 = time.time()
    rc, out, err = run([str(VENV_DIR / "bin" / "pip"), "install", f"vllm=={VLLM_VERSION}",
                        "--no-deps", "--timeout", "60"], 1200)
    ok_main = rc == 0
    record("vllm_install", ok_main=ok_main, rc=rc, err=tail(err, 800),
           seconds=round(time.time() - t0, 1))
    for dep in VLLM_DEPS:
        rc_d, out_d, err_d = run([str(VENV_DIR / "bin" / "pip"), "install", dep, "--timeout", "60"], 300)
        record("vllm_install", **{f"dep_{dep.split('==')[0].replace('-', '_')}": rc_d == 0})
        if rc_d != 0:
            marker("vllm_install", "dep_fail", dep=dep)
    ok = ok_main
    marker("vllm_install", "done" if ok else "fail")
    return ok


def resolve_base_model() -> str | None:
    for cand in BASE_MODEL_CANDIDATES:
        if Path(cand).exists():
            return cand
    models = Path("/data/magnus/models")
    if models.exists():
        for p in sorted(models.iterdir()):
            if p.name.startswith("Qwen3.5-9B") and "sft" not in p.name:
                return str(p)
    return None


E2E_SCRIPT = r'''
import json, time, sys
model_path = sys.argv[1]
t0 = time.time()
from vllm import LLM, SamplingParams
print("=== R3-0 STAGE vllm_import done seconds=%.1f ===" % (time.time()-t0), flush=True)
t1 = time.time()
llm = LLM(model=model_path, max_model_len=4096, dtype="bfloat16", gpu_memory_utilization=0.85)
print("=== R3-0 STAGE vllm_load done seconds=%.1f ===" % (time.time()-t1), flush=True)
t2 = time.time()
out = llm.generate(["1+1等于几?用一句话回答。"], SamplingParams(max_tokens=20, temperature=0.0))
gen_s = time.time()-t2
text = out[0].outputs[0].text
ntok = len(out[0].outputs[0].token_ids)
print("=== R3-0 STAGE vllm_generate done seconds=%.2f tokens=%d ===" % (gen_s, ntok), flush=True)
print("=== R3-0 GEN_TEXT begin ===", flush=True)
print(text, flush=True)
print("=== R3-0 GEN_TEXT end ===", flush=True)
'''


def stage_vllm_import() -> bool:
    t0 = time.time()
    repair_map = {"cloudpickle": "cloudpickle", "pydantic": "pydantic",
                  "typing_inspection": "typing-inspection", "annotated_types": "annotated-types",
                  "tiktoken": "tiktoken", "jsonschema": "jsonschema", "einops": "einops"}
    rounds = 0
    while rounds <= 2:
        rc, out, err = run([str(VENV_DIR / "bin" / "python"), "-c",
                            "import vllm; print(vllm.__version__)"], 600)
        record("vllm_import", rc=rc, err=tail(err, 1500), out=tail(out, 300),
               seconds=round(time.time() - t0, 1), rounds=rounds)
        if rc == 0:
            marker("vllm_import", "done", rounds=rounds)
            return True
        miss = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", err)
        if miss and rounds < 2 and miss.group(1).split(".")[0] in repair_map:
            pkg = repair_map[miss.group(1).split(".")[0]]
            marker("vllm_import", "repair", pkg=pkg, rounds=rounds)
            run([str(VENV_DIR / "bin" / "pip"), "install", pkg, "--timeout", "60"], 300)
            rounds += 1
            continue
        marker("vllm_import", "fail", rounds=rounds)
        return False
    return False


def stage_vllm_load_generate(model_path: str) -> bool:
    script = WORK_DIR / "e2e.py"
    script.write_text(E2E_SCRIPT, encoding="utf-8")
    t0 = time.time()
    env = dict(os.environ)
    env.update({"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
                "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    rc, out, err = run([str(VENV_DIR / "bin" / "python"), str(script), model_path], 2400, env=env)
    load_m = re.search(r"vllm_load done seconds=([\d.]+)", out)
    gen_m = re.search(r"vllm_generate done seconds=([\d.]+) tokens=(\d+)", out)
    gen_text = ""
    if "R3-0 GEN_TEXT begin" in out:
        gen_text = out.split("=== R3-0 GEN_TEXT begin ===")[1].split("=== R3-0 GEN_TEXT end ===")[0].strip()
    import_ok = "vllm_import done" in out
    record("vllm_load_generate", rc=rc, import_stage_s=load_m.group(1) if load_m else None,
           load_s=load_m.group(1) if load_m else None, gen_s=gen_m.group(1) if gen_m else None,
           gen_tokens=int(gen_m.group(2)) if gen_m else None, gen_text=gen_text[:500],
           err=tail(err, 2500), out_tail=tail(out, 2500), seconds=round(time.time() - t0, 1))
    ok = rc == 0 and gen_m is not None and bool(gen_text)
    marker("vllm_load_generate", "done" if ok else "fail")
    return ok


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    stage_env()
    ok_venv = stage_venv()
    ok_torch = stage_torch_install() if ok_venv else False
    if ok_torch:
        stage_cuda_matmul()
        stage_inductor_api()
    ok_vllm = stage_vllm_install() if ok_torch else False
    ok_import = stage_vllm_import() if ok_vllm else False
    model = resolve_base_model()
    MATRIX["base_model"] = model
    ok_gen = False
    if ok_import and model:
        ok_gen = stage_vllm_load_generate(model)
    elif not model:
        record("vllm_load_generate", skipped="no base model path found")
    verdict = {
        "torch27_cuda": ok_torch, "vllm_import": ok_import,
        "vllm_generate": ok_gen,
        "unified_env": bool(ok_torch and ok_vllm and ok_gen),
    }
    MATRIX["verdict"] = verdict
    (OUT_DIR / "receipt.json").write_text(json.dumps(MATRIX, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**R3-0_MATRIX** " + json.dumps(MATRIX, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
