"""
0828-verify 任务 2:教师 Qwen3.8-27B vLLM 单卡部署冒烟。
- 单卡 vLLM,禁止 TP(集群 NCCL busbw 1.6 GiB/s,TP 必卡)
- 27B bf16 ≈54GB,max-model-len 32768, gpu-memory-utilization 0.92
- 冒烟:3 道种子题(矢量恒等式 / 一阶线性 ODE / SymPy 化简),各 1 条带思考完整解答
- 加测(决议 R2):同题 effort medium vs low 各采一次,记录链长与终值正确率
- 离线形态:job 内 LLM.generate(冒烟阶段不需要常驻 service)

用法(在集群容器内):
  python 0828_teacher_smoke.py --model /data/magnus/models/Qwen3.8-27B-20260828
"""
import argparse, json, time, os, re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/magnus/models/Qwen3.8-27B-20260828")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--out", default="/data/magnus/smoke-0828/teacher")
    return p.parse_args()


SEED_PROBLEMS = [
    {
        "id": "vec-curl-001",
        "type": "vector_identity",
        "prompt": (
            "设 F(x,y,z) = (x^2 y, y^2 z, z^2 x) 是一个向量场。"
            "验证恒等式 ∇·(∇×F) = 0:先写出 ∇×F 的三个分量,"
            "再求其散度并化简到 0。请给出完整的推导过程。"
        ),
    },
    {
        "id": "ode-001",
        "type": "first_order_linear_ode",
        "prompt": (
            "求解一阶线性常微分方程 y' + 2 y = e^{-x},初值 y(0) = 1。"
            "用积分因子法推导通解,再代入初值定出常数,给出特解并验证。"
        ),
    },
    {
        "id": "sympy-simplify-001",
        "type": "sympy_simplify",
        "prompt": (
            "化简表达式 (sin(x))^4 - (cos(x))^4 + cos(2x)。"
            "要求:用三角恒等式化简到最简形式,并给出化简依据;"
            "最后用 SymPy 的 simplify/trigsimp 验证结果一致。"
        ),
    },
]

# 官方推荐 thinking 采样参数
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"=== loading model {args.model} ===")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        tensor_parallel_size=1,  # 禁 TP
    )
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    results = {"load_seconds": load_s, "problems": {}}

    for prob in SEED_PROBLEMS:
        print(f"\n=== {prob['id']} ===")
        # 构造 chat messages:user 角色
        messages = [{"role": "user", "content": prob["prompt"]}]
        text = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        # 确认 <think> 段在渲染结果中
        results["problems"][prob["id"]] = {"type": prob["type"], "rendered_has_think": "<think>" in text}

        for effort in ["medium", "low"]:
            print(f"  effort={effort}")
            # Qwen3.8 的 reasoning_effort 是 chat-template 参数,不是 SamplingParams 参数。
            # 模板会把 effort 说明写入 system prompt,从而控制思考深度。
            effort_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                chat_template_kwargs={"enable_thinking": True, "reasoning_effort": effort},
            )
            sp = SamplingParams(
                max_tokens=4096,
                **SAMPLING,
            )
            t1 = time.time()
            out = llm.generate([effort_text], sp)
            gen_s = time.time() - t1
            gen = out[0].outputs[0]
            full = gen.text
            n_tokens = len(gen.token_ids)
            # 拆 think / answer
            think_m = re.search(r"<think>(.*?)</think>", full, re.S)
            think_txt = think_m.group(1) if think_m else ""
            answer_txt = full[think_m.end():] if think_m else full
            entry = {
                "effort": effort,
                "generation_seconds": round(gen_s, 2),
                "tokens": n_tokens,
                "tokens_per_sec": round(n_tokens / gen_s, 1) if gen_s > 0 else None,
                "first_token_latency_s": None,  # 离线接口不给逐 token 时间;部署冒烟用总吞吐
                "think_chars": len(think_txt),
                "think_tokens_est": len(think_txt) // 2,  # 粗略估计
                "answer_preview": answer_txt[:300],
            }
            results["problems"][prob["id"]][effort] = entry
            print(f"    tokens={n_tokens}, {entry['tokens_per_sec']} tok/s, think={entry['think_chars']} chars")
            print(f"    answer preview: {answer_txt[:120]!r}")

    out_path = os.path.join(args.out, "teacher_smoke_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== results written to {out_path} ===")


if __name__ == "__main__":
    main()
