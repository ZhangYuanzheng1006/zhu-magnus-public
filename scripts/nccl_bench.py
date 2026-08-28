# -*- coding: utf-8 -*-
"""NCCL allreduce busbw 单项基准（容器内，必须整文件作为 __main__ 运行）。
world_size=2；用 spawn + 正确的 __main__ 保护，避免子进程重放顶层代码。
"""
import json
import os
import time


def worker(rank):
    import torch
    import torch.distributed as dist
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=2)
    out = {}
    for mb in (8, 128, 512):
        x = torch.ones(mb * 2**20 // 4, device=f"cuda:{rank}")
        for _ in range(3):
            dist.all_reduce(x)                        # warmup
        dist.barrier()
        torch.cuda.synchronize(rank)
        t0 = time.perf_counter(); reps = 10
        for _ in range(reps):
            dist.all_reduce(x)
        dist.barrier()
        torch.cuda.synchronize(rank)
        dt = (time.perf_counter() - t0) / reps
        algbw = mb * 2**20 / dt / 2**30
        busbw = algbw                                  # ring allreduce n=2 → busbw == algbw
        if rank == 0:
            out[f"allreduce_{mb}MB"] = round(busbw, 1)
            print(f"allreduce {mb:>4} MB : algbw {algbw:.1f} | busbw {busbw:.1f} GiB/s",
                  flush=True)
    dist.destroy_process_group()
    if rank == 0:
        print("**NCCL_JSON** " + json.dumps(out), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29518")
    os.environ["NCCL_SHM_DISABLE"] = "1"       # containall 下 /dev/shm 受限
    os.environ["NCCL_DEBUG"] = "WARN"

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r,), daemon=False) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=240)
    hung = [p.is_alive() for p in procs]
    if any(hung):
        for p in procs:
            p.terminate()
        print(f"[nccl] workers hung: {hung} -> terminated", flush=True)
    else:
        print(f"[nccl] exitcodes: {[p.exitcode for p in procs]}", flush=True)
