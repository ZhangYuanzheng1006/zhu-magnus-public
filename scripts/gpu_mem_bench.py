# -*- coding: utf-8 -*-
"""LLM 相关硬件带宽/延时基准（容器内运行，目标镜像 pytorch/pytorch:2.5.1-cuda12.4）。

测项：
 A. 拓扑与身份：nvidia-smi 全量/topo/nvlink 状态、PCIe 链路档位、CPU/NUMA、iGPU 探测
 B. CPU 内存带宽（多线程 torch copy + 单线程 numpy copy）、CPU FP32 GEMM 算力
 C. H2D/D2H 带宽（pinned vs pageable）+ 小包延时；单卡 D2D 带宽
 D. 双卡 P2P：can_access_peer、单向/双向并发拷贝带宽、P2P 小包延时（无 P2P 则走 host 中转对照）
 E. NCCL allreduce busbw（真实 LLM 通信路径）
最终输出一行 **SUMMARY_JSON** {...} 便于本地解析。
运行预算 ≈ 5 分钟。
"""
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

T0 = time.time()
RESULTS = {}
GB = float(2**30)


def sec(title):
    print("\n" + "=" * 72, flush=True)
    print(f"[{time.time()-T0:7.1f}s] {title}", flush=True)
    print("=" * 72, flush=True)


def sh(cmd, label=None):
    """跑 shell 命令并原样打印输出；命令缺失/失败不致命。"""
    print(f"\n--- $ {label or cmd}", flush=True)
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
        text = ((out.stdout or "") + (out.stderr or "")).strip()
        print(text[:6000], flush=True)
        return text
    except Exception as e:
        print(f"[unavailable: {type(e).__name__}: {e}]", flush=True)
        return ""


def timed(fn, repeat=5):
    """best-of repeat，含 synchronize。"""
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


# ════════════════════ A. 拓扑 / NVLink / PCIe / CPU 身份 ════════════════════
sec("A. 拓扑 / NVLink / PCIe / CPU")

sh("nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total,"
   "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max "
   "--format=csv", label="PCIe link per GPU")
topo = sh("nvidia-smi topo -m", label="*** NVLink 判定最直接证据：topo 矩阵 ***")
sh("for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do echo \"== GPU $i ==\";"
   " nvidia-smi nvlink --status -i $i 2>&1 | head -6; done", label="nvlink status per GPU")
sh("lscpu", label="CPU info")
sh("free -g && grep -E 'MemTotal|MemAvailable' /proc/meminfo", label="memory")
sh("dmidecode -t memory 2>/dev/null | grep -E 'Maximum Capacity|Speed:|Type:'"
   " | sort | uniq -c | head -12 || echo 'dmidecode needs root - skipped'", label="RAM type/speed")
sh("lspci 2>/dev/null | grep -iE 'vga|3d controller|display' || echo '(no lspci or no iGPU found)'",
   label="iGPU / display adapters")
sh("cat /sys/devices/system/node/node*/cpulist 2>/dev/null || echo '(single node)'", label="NUMA cpu lists")

# 只检查矩阵数据行（以 GPU<digit> 开头的行），避免匹配到图例文本里的字面 "NV#"
matrix_rows = [ln for ln in topo.splitlines() if ln.lstrip().startswith("GPU")]
has_nvlink_row = any("NV" in ln and "\tNV" in ln for ln in matrix_rows)
nvl_cli = sh("nvidia-smi nvlink -i 0 --status 2>&1 | head -3", label="nvlink authoritative check")
if has_nvlink_row:
    nvl_verdict = "matrix rows contain NV* links → NVLink PRESENT"
elif "inActive" in nvl_cli or "Unable" in nvl_cli:
    nvl_verdict = "matrix rows show PCIe-class connection AND nvlink reports links inactive → NO usable NVLink"
else:
    nvl_verdict = f"matrix rows show {['PCIe-class']} connection → NO NVLink"
print(f"\n[verdict from topo -m] {nvl_verdict}", flush=True)
RESULTS["nvlink_topo_verdict"] = nvl_verdict

n_gpus = torch.cuda.device_count()
RESULTS["gpu_count"] = n_gpus
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    RESULTS[f"gpu{i}"] = {"name": p.name, "vram_GiB": round(p.total_memory / GB, 1)}

# ════════════════════ B. CPU 内存带宽 / 算力 ════════════════════
sec("B. CPU 内存带宽与算力")
torch.set_num_threads(os.cpu_count() or 8)
print(f"torch threads = {torch.get_num_threads()}", flush=True)

for gb in (0.5, 2.0):
    n_el = int(gb * GB / 4)
    a = torch.empty(n_el); b = torch.empty(n_el); a.normal_()
    best = float("inf")
    for _ in range(4):
        t0 = time.perf_counter(); b.copy_(a); best = min(best, time.perf_counter() - t0)
    eff = n_el * 4 * 2 / best / 1e9          # copy = 读 size + 写 size
    print(f"CPU bandwidth torch multi-thread {gb:>4} GiB chunk : "
          f"{n_el*4/best/1e9:.1f} GB/s payload | {eff:.1f} GB/s effective(R+W)", flush=True)
    RESULTS[f"cpu_bw_mt_{gb}GiB_eff_GBs"] = round(eff)

na = np.empty(int(0.25 * GB / 4), dtype=np.float32)
na.fill(1.0)
nb = np.empty_like(na)
t0 = time.perf_counter()
for _ in range(3):
    np.copyto(nb, na)
st = na.nbytes * 2 / ((time.perf_counter() - t0) / 3) / 1e9
del na, nb
print(f"CPU bandwidth numpy single-thread 256 MiB       : ~{st:.1f} GB/s effective(R+W)", flush=True)
RESULTS["cpu_bw_single_thread_256MiB_eff_GBs"] = round(st)

k = 3072
am = torch.randn(k, k); bm = torch.randn(k, k)
am @ bm                                             # warmup + raise clocks
t0 = time.perf_counter()
reps = 3
for _ in range(reps):
    cm = am @ bm
gflops = 2 * k**3 * reps / (time.perf_counter() - t0) / 1e9
del am, bm, cm
print(f"CPU FP32 GEMM ({torch.get_num_threads()} threads) : {gflops:.0f} GFLOPS", flush=True)
RESULTS["cpu_fp32_gflops"] = round(gflops)

# ════════════════════ C. 主机<->显存 与 单卡 D2D ════════════════════
sec("C. Host<->Device (GPU0) 带宽/延时 与 单卡 D2D")
GIB_PAYLOAD = 1                                        # 每次搬运的负载大小
n_el = int(GIB_PAYLOAD * GB / 4)
host_pinned = torch.empty(n_el, pin_memory=True); host_pinned.normal_()
host_pageable = torch.randn(n_el)
gpu_buf = torch.empty(n_el, device="cuda:0")

t_h2d = timed(lambda: gpu_buf.copy_(host_pinned))
t_d2h = timed(lambda: host_pinned.copy_(gpu_buf))
t_h2d_pg = timed(lambda: gpu_buf.copy_(host_pageable), repeat=3)
results_h2d = GIB_PAYLOAD * GB / t_h2d / GB
results_d2h = GIB_PAYLOAD * GB / t_d2h / GB
results_pg = GIB_PAYLOAD * GB / t_h2d_pg / GB
print(f"H2D pinned   {GIB_PAYLOAD} GiB : {results_h2d:.1f} GiB/s", flush=True)
print(f"D2H pinned   {GIB_PAYLOAD} GiB : {results_d2h:.1f} GiB/s", flush=True)
print(f"H2D pageable {GIB_PAYLOAD} GiB : {results_pg:.1f} GiB/s", flush=True)
RESULTS.update(h2d_pinned_GiBs=round(results_h2d, 1),
               d2h_pinned_GiBs=round(results_d2h, 1),
               h2d_pageable_GiBs=round(results_pg, 1))

tiny_f = torch.zeros(1, pin_memory=True)
tiny_t = torch.zeros(1, device="cuda:0")
tiny_t.copy_(tiny_f)
N_LAT = 1000
t0 = time.perf_counter()
for _ in range(N_LAT):
    tiny_t.copy_(tiny_f)
torch.cuda.synchronize()
lat_us = (time.perf_counter() - t0) / N_LAT * 1e6
print(f"H2D small-packet latency (4 B pinned, avg {N_LAT}) : {lat_us:.1f} us", flush=True)
RESULTS["h2d_latency_us"] = round(lat_us, 1)

n_g1 = int(GB / 4)
g_src = torch.ones(n_g1, device="cuda:0"); g_dst = torch.empty(n_g1, device="cuda:0")
t_d2d = timed(lambda: g_dst.copy_(g_src))
print(f"D2D intra-GPU 1 GiB copy : payload {GB/t_d2d/GB:.1f} GiB/s | effective(R+W) "
      f"{2*GB/t_d2d/GB:.1f} GiB/s", flush=True)
RESULTS["d2d_payload_GiBs"] = round(GB / t_d2d / GB, 1)
RESULTS["d2d_effective_rw_GiBs"] = round(2 * GB / t_d2d / GB, 1)
del g_src, g_dst, gpu_buf, host_pinned, host_pageable
torch.cuda.empty_cache()

# ════════════════════ D. 双卡 P2P ════════════════════
sec("D. GPU0 <-> GPU1 P2P（NVLink 实证交叉验证）")
peer01 = torch.cuda.can_device_access_peer(0, 1)
peer10 = torch.cuda.can_device_access_peer(1, 0)
print(f"can_device_access_peer(0->1)={peer01}, (1->0)={peer10}", flush=True)
RESULTS["can_access_peer"] = bool(peer01 and peer10)

pa = torch.ones(int(GB / 4), device="cuda:0")
pb = torch.empty(int(GB / 4), device="cuda:1")

if peer01:
    t_uni = timed(lambda: pb.copy_(pa), 8)
    uni = GB / t_uni / GB
    print(f"P2P unidirectional 0->1, 1 GiB : {uni:.1f} GiB/s", flush=True)
    RESULTS["p2p_unidirectional_GiBs"] = round(uni, 1)

    s0 = torch.cuda.Stream(device=0); s1 = torch.cuda.Stream(device=1)
    pa2 = torch.ones(int(GB / 4), device="cuda:1")            # 用作 1->0 方向源

    def duplex():
        with torch.cuda.stream(s0):
            pb.copy_(pa)                                       # 0 -> 1
        with torch.cuda.stream(s1):
            pa.copy_(pa2)                                      # 1 -> 0
        s0.synchronize(); s1.synchronize()

    t_dup = timed(duplex, 8)
    agg = 2 * GB / t_dup / GB
    print(f"P2P bidirectional concurrent   : {agg:.1f} GiB/s aggregate ({agg/2:.1f} each way)",
          flush=True)
    RESULTS["p2p_bidirectional_aggregate_GiBs"] = round(agg, 1)

    tiny0 = torch.zeros(1, device="cuda:0"); tiny1 = torch.zeros(1, device="cuda:1")
    tiny0.copy_(tiny1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_LAT):
        tiny0.copy_(tiny1)
    torch.cuda.synchronize()
    p2p_lat = (time.perf_counter() - t0) / N_LAT * 1e6
    print(f"P2P small-packet latency (4 B, avg {N_LAT}) : {p2p_lat:.1f} us", flush=True)
    RESULTS["p2p_latency_us"] = round(p2p_lat, 1)
else:
    def staged():
        tmp = pa.to("cpu", non_blocking=False)
        pb.copy_(tmp)
    t_st = timed(staged, 3)
    st_rate = GB / t_st / GB
    print(f"P2P unavailable → staged via host 1 GiB : {st_rate:.1f} GiB/s (fallback path)", flush=True)
    RESULTS["p2p_via_host_GiBs"] = round(st_rate, 1)

del pa, pb
try:
    del pa2
except NameError:
    pass
torch.cuda.empty_cache()

# ════════════════════ E. NCCL allreduce ════════════════════
sec("E. NCCL allreduce busbw (world=2, 单解释器 mp.spawn)")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
# containall 容器里 /dev/shm 受限会挂死 NCCL 初始化；禁 SHM 走 P2P/NET 传输
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"


def _nccl_worker(rank):
    import torch.distributed as dist
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=2)
    for mb in (8, 128, 512):
        x = torch.ones(mb * 2**20 // 4, device=f"cuda:{rank}")
        for _ in range(3):
            dist.all_reduce(x)                       # warmup
        dist.barrier(); torch.cuda.synchronize(rank)
        t0 = time.perf_counter(); reps = 10
        for _ in range(reps):
            dist.all_reduce(x)
        dist.barrier(); torch.cuda.synchronize(rank)
        dt = (time.perf_counter() - t0) / reps
        algbw = mb * 2**20 / dt / 2**30
        busbw = algbw                                 # ring n=2: algbw*2*(n-1)/n = algbw
        if rank == 0:
            RESULTS[f"nccl_allreduce_{mb}MB_busbw_GiBs"] = round(busbw, 1)
            print(f"allreduce {mb:>4} MB : algbw {algbw:.1f} GiB/s | busbw {busbw:.1f} GiB/s",
                  flush=True)
    dist.destroy_process_group()


import torch.multiprocessing as mp
try:
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_nccl_worker, args=(r,), daemon=False) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=240)
    hung = [p.is_alive() for p in procs]
    if any(hung):
        for p in procs:
            p.terminate()
        print(f"NCCL workers hung: {hung} → terminated", flush=True)
        RESULTS["nccl_error"] = "workers timed out / terminated"
    else:
        print(f"nccl worker exitcodes: {[p.exitcode for p in procs]}", flush=True)
except Exception as e:
    print(f"NCCL bench failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    RESULTS["nccl_error"] = f"{type(e).__name__}: {str(e)[:200]}"

# ════════════════════ 总结 ════════════════════
sec("SUMMARY")
print("**SUMMARY_JSON** " + json.dumps(RESULTS, ensure_ascii=False, default=str), flush=True)
print(f"\nTotal bench wall time: {time.time() - T0:.0f}s", flush=True)
