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

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

VERSION = "r3-0-torch27-probe-v4"
VENV_DIR = Path("/dev/shm/r3-0/venv")
WHEEL_DIR = Path("/tmp/r3-0/wheels")
WORK_DIR = Path("/tmp/r3-0")
OUT_DIR = Path(os.environ.get("R3_OUT", "/data/magnus/closedloop-0828/r3-0-torch27-v3"))
BASE_MODEL_CANDIDATES = [
    "/data/magnus/models/Qwen3.5-9B-20260828",
]
TORCH_VERSION = "2.7.1"
VLLM_VERSION = "0.22.1"
# v3 finding: the vllm 0.22.1 wheel currently served by the default PyPI
# mirror is CUDA-13-linked (import vllm._C -> libcudart.so.13 missing), while
# the R2-0-era wheel of the same version loaded its _C on cu124. v4 hunts for
# a CUDA-12-linked wheel: current mirror 0.22.1/0.23.0, then the official
# per-release wheel index. First candidate whose _C imports wins the e2e.
VLLM_CANDIDATES = [
    ("pypi_0.22.1", [], "vllm==0.22.1"),
    ("pypi_0.23.0", [], "vllm==0.23.0"),
    ("wheels_vllm_ai_0.22.1", ["--index-url", "https://wheels.vllm.ai/0.22.1/"], "vllm==0.22.1"),
    ("wheels_vllm_ai_0.23.0", ["--index-url", "https://wheels.vllm.ai/0.23.0/"], "vllm==0.23.0"),
]
# Pure-python / pinned deps proven necessary by the R2-0 v2..v8 probe series.
VLLM_DEPS = [
    "cloudpickle",
    "pydantic==2.13.5",
    "pydantic-core==2.46.5",
    "typing-inspection==0.4.2",
    "annotated-types==0.7.0",
]
# v2 lesson: the full torch stack needs >20 min on the slow-write /tmp FS and
# switching indexes busted the pip cache. v3 downloads every wheel once (phase
# A) and installs from the local dir (phase B), with generous budgets.
TORCH_ATTEMPTS = [
    ("download_then_local_install", [], 2400, 2400),
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


def stream_cmd(cmd: list[str], timeout: int, *, env: dict[str, str] | None = None,
               heartbeat_s: int = 45) -> tuple[int, str, str]:
    """Long installs: stream pip lines live and print a heartbeat so the job
    log never goes silent (v1 was slurm-CANCELLED mid silent download)."""
    import threading
    t0 = time.time()
    stop = threading.Event()
    marker("heartbeat", "start", every_s=heartbeat_s)

    def beat() -> None:
        n = 0
        while not stop.wait(heartbeat_s):
            n += 1
            print(f"[heartbeat] alive_s={int(time.time() - t0)} beat={n}", flush=True)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    out_lines: list[str] = []
    err_lines: list[str] = []
    rc = 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env)
        import threading as th2

        def pump(src, sink: list[str], tag: str) -> None:
            for line in src:
                sink.append(line)
                if tag == "out" and len(sink) % 5 == 0:
                    print(f"pip| {line.strip()[:160]}", flush=True)

        t_out = th2.Thread(target=pump, args=(proc.stdout, out_lines, "out"), daemon=True)
        t_err = th2.Thread(target=pump, args=(proc.stderr, err_lines, "err"), daemon=True)
        t_out.start(); t_err.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
        t_out.join(5); t_err.join(5)
    except Exception as exc:  # noqa: BLE001
        return 125, "", f"{type(exc).__name__}: {exc}"
    finally:
        stop.set(); thread.join(2)
        print(f"--- streamed cmd took {time.time() - t0:.1f}s rc={rc}: {' '.join(cmd[:6])} ...", flush=True)
    return rc, "".join(out_lines), "".join(err_lines)


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
    """Phase A: download all wheels to WHEEL_DIR. Phase B: install from the
    local dir only (--no-index --find-links), so network flakiness and cache
    key changes cannot poison the install; /dev/shm venv dodges slow writes."""
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = "/tmp/r3-0/pipcache"
    env["PIP_NO_COMPILE"] = "1"
    name, extra, dl_timeout, inst_timeout = TORCH_ATTEMPTS[0]
    t0 = time.time()
    marker("torch_download", "start")
    dl_cmd = [str(VENV_DIR / "bin" / "pip"), "download", f"torch=={TORCH_VERSION}",
              "-d", str(WHEEL_DIR), "--timeout", "60", "--retries", "3",
              "--progress-bar", "off", *extra]
    rc, out, err = stream_cmd(dl_cmd, dl_timeout, env=env)
    record("torch_download", rc=rc, seconds=round(time.time() - t0, 1), err=tail(err, 800),
           wheels=[p.name for p in WHEEL_DIR.glob("*.whl")][:20] if WHEEL_DIR.exists() else [])
    marker("torch_download", "done" if rc == 0 else "fail", seconds=round(time.time() - t0, 1))
    if rc != 0:
        MATRIX["torch_source"] = None
        marker("torch_install", "fail", stage="download")
        return False
    t1 = time.time()
    marker("torch_install", "start", source="local_wheels")
    inst_cmd = [str(VENV_DIR / "bin" / "pip"), "install", "--no-index",
                "--find-links", str(WHEEL_DIR), f"torch=={TORCH_VERSION}",
                "--progress-bar", "off"]
    rc, out, err = stream_cmd(inst_cmd, inst_timeout, env=env)
    if rc != 0:
        record("torch_install", rc=rc, err=tail(err, 800), seconds=round(time.time() - t1, 1))
        marker("torch_install", "fail", stage="install", seconds=round(time.time() - t1, 1))
        MATRIX["torch_source"] = None
        return False
    ok, detail = torch_verify()
    record("torch_install", ok=ok, detail=detail[:1200], seconds=round(time.time() - t1, 1))
    MATRIX["torch_source"] = name
    marker("torch_install", "done" if ok else "verify_fail", detail=detail[:200])
    return ok


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


def stage_vllm_deps() -> bool:
    t0 = time.time()
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = "/tmp/r3-0/pipcache"
    env["PIP_NO_COMPILE"] = "1"
    rc, out, err = stream_cmd([str(VENV_DIR / "bin" / "pip"), "install", *VLLM_DEPS,
                               "--timeout", "60", "--progress-bar", "off"], 600, env=env)
    record("vllm_deps", rc=rc, seconds=round(time.time() - t0, 1), err=tail(err, 500))
    marker("vllm_deps", "done" if rc == 0 else "fail")
    return rc == 0


def wheel_provenance(wheel_dir: Path) -> list[dict[str, Any]]:
    info = []
    for p in sorted(wheel_dir.glob("*.whl")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        info.append({"file": p.name, "bytes": p.stat().st_size, "sha256": h})
    return info


def stage_vllm_hunt() -> tuple[bool, str | None]:
    """Try each candidate wheel; first one whose compiled _C imports on this
    driver/runtime goes straight to the 9B e2e. Everything is fingerprinted."""
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = "/tmp/r3-0/pipcache"
    env["PIP_NO_COMPILE"] = "1"
    c_probe = "import vllm._C; import vllm; print('vllm', vllm.__version__, '_C OK')"
    for tag, extra, spec in VLLM_CANDIDATES:
        wheel_dir = WORK_DIR / f"vllm-wheels-{tag}"
        t0 = time.time()
        marker("vllm_hunt", "download", tag=tag)
        dl = [str(VENV_DIR / "bin" / "pip"), "download", spec, "--no-deps",
              "-d", str(wheel_dir), "--timeout", "60", "--progress-bar", "off", *extra]
        rc, out, err = stream_cmd(dl, 900, env=env)
        prov = wheel_provenance(wheel_dir) if wheel_dir.exists() else []
        record("vllm_hunt", **{f"{tag}_download_rc": rc, f"{tag}_wheels": prov,
                               f"{tag}_download_s": round(time.time() - t0, 1),
                               f"{tag}_download_err": tail(err, 500)})
        if rc != 0 or not prov:
            marker("vllm_hunt", "download_fail", tag=tag)
            continue
        wheel_file = str(wheel_dir / prov[0]["file"])
        rc, out, err = stream_cmd([str(VENV_DIR / "bin" / "pip"), "install", "--no-deps",
                                   wheel_file, "--progress-bar", "off"], 600, env=env)
        record("vllm_hunt", **{f"{tag}_install_rc": rc})
        if rc != 0:
            marker("vllm_hunt", "install_fail", tag=tag)
            continue
        rc, out, err = run([str(VENV_DIR / "bin" / "python"), "-c", c_probe], 300, env=env)
        record("vllm_hunt", **{f"{tag}_c_probe_rc": rc, f"{tag}_c_probe_out": tail(out, 200),
                               f"{tag}_c_probe_err": tail(err, 1200)})
        if rc != 0:
            kind = "cuda13_linked" if "libcudart.so.13" in err else "c_probe_other"
            marker("vllm_hunt", "c_probe_fail", tag=tag, kind=kind)
            record("vllm_hunt", **{f"{tag}_fail_kind": kind})
            # uninstall so the next candidate's files are authoritative
            run([str(VENV_DIR / "bin" / "pip"), "uninstall", "-y", "vllm"], 120, env=env)
            continue
        marker("vllm_hunt", "c_probe_ok", tag=tag)
        return True, tag
    marker("vllm_hunt", "exhausted")
    return False, None


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


def start_gpu_keepalive() -> subprocess.Popen | None:
    """v1 was slurm-CANCELLED ~21 min in with the GPU allocated but untouched
    during silent pip downloads. Keep a tiny CUDA workload alive throughout."""
    script = (
        "import torch, time\n"
        "while True:\n"
        "    try:\n"
        "        a = torch.randn(256, 256, device='cuda'); b = torch.randn(256, 256, device='cuda')\n"
        "        c = (a @ b).sum().item(); open('/tmp/r3-0/keepalive.txt', 'w').write(str(time.time()))\n"
        "    except Exception:\n"
        "        pass\n"
        "    time.sleep(30)\n"
    )
    p = subprocess.Popen(["python3", "-c", script], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    marker("keepalive", "start", pid=p.pid)
    return p


def pick_venv_dir() -> Path:
    """Prefer RAM-backed /dev/shm (v2: unpacking ~3.5GB of wheels onto the
    slow-write disk FS blew every per-attempt budget); fall back to /tmp."""
    try:
        free = shutil.disk_usage("/dev/shm").free
    except Exception:  # noqa: BLE001
        free = 0
    if free >= 12 * 2**30:
        return Path("/dev/shm/r3-0/venv")
    return Path("/tmp/r3-0/venv")


def main() -> int:
    global VENV_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    VENV_DIR = pick_venv_dir()
    MATRIX["venv_dir"] = str(VENV_DIR)
    keepalive = start_gpu_keepalive()
    try:
        stage_env()
        ok_venv = stage_venv()
        ok_torch = stage_torch_install() if ok_venv else False
        if ok_torch:
            stage_cuda_matmul()
            stage_inductor_api()
        ok_c = False
        won_tag = None
        if ok_torch:
            ok_deps = stage_vllm_deps()
            if ok_deps:
                ok_c, won_tag = stage_vllm_hunt()
        MATRIX["vllm_wheel_tag"] = won_tag
        model = resolve_base_model()
        MATRIX["base_model"] = model
        ok_gen = False
        if ok_c and model:
            ok_gen = stage_vllm_load_generate(model)
        elif not model:
            record("vllm_load_generate", skipped="no base model path found")
    finally:
        if keepalive:
            keepalive.terminate()
            marker("keepalive", "stop", pid=keepalive.pid)
    verdict = {
        "torch27_cuda": ok_torch, "vllm_import": ok_c,
        "vllm_generate": ok_gen,
        "unified_env": bool(ok_torch and ok_c and ok_gen),
    }
    MATRIX["verdict"] = verdict
    (OUT_DIR / "receipt.json").write_text(json.dumps(MATRIX, ensure_ascii=False, indent=2), encoding="utf-8")
    print("**R3-0_MATRIX** " + json.dumps(MATRIX, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
