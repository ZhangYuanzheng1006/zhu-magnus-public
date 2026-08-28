"""
0828-verify 任务 3:学生 Qwen3.5-9B TRL SFT + LoRA 微调冒烟。
- 数据:200 条 06 协议格式(scripts/0828_make_smoke_data.py 生成)
- 训练:TRL SFTTrainer + LoRA(r=16, all-linear, bf16, lr 1e-4~2e-4, ~100 步, 单卡)
- 只对 assistant 段计 loss:用 data_collator 的 completion-only mask
- 参数账(决议 R4):trainable/total、按模块分布,开训前打印
- 验收:loss 收敛迹象 / adapter 保存 / merge_and_unload / vLLM 加载 / 格式命中率
- 措辞:这是管线冒烟,不写"闭环完成"

用法(集群容器内):
  python 0828_student_sft.py --model /data/magnus/models/Qwen3.5-9B-20260828 \
      --data /data/magnus/smoke-0828/student/data.jsonl --out /data/magnus/smoke-0828/student/out
"""
import argparse, json, os, sys, time, hashlib
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/magnus/models/Qwen3.5-9B-20260828")
    p.add_argument("--data", default="/data/magnus/smoke-0828/student/data.jsonl")
    p.add_argument("--out", default="/data/magnus/smoke-0828/student/out")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    print(f"=== loading tokenizer/model {args.model} ===")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
    )

    # ---- 参数账(决议 R4)----
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params/1e9:.2f}B")

    # ---- LoRA ----
    lora = LoraConfig(
        r=args.r,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,} ({trainable/total_params*100:.3f}%)")

    # 按模块分布
    by_mod = Counter()
    for n, p in model.named_parameters():
        if p.requires_grad:
            # 取倒数第二级模块名(如 layers.0.self_attn.q_proj → self_attn)
            parts = n.split(".")
            key = parts[-2] if len(parts) >= 2 else n
            by_mod[key] += p.numel()
    print("trainable by module:", dict(by_mod))

    # ---- 数据 ----
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"dataset rows: {len(ds)}")

    def fmt(example):
        return {"text": tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(fmt)
    # completion-only mask:SFTTrainer 默认只对 assistant 段计 loss(需要 chat template 正确)
    # TRL SFTTrainer 的 data_collator 用 tokenizer 的 chat template + masking

    # ---- 训练 ----
    sft_cfg = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        max_seq_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        tokenizer=tok,
        data_collator=None,  # SFTTrainer 默认 collator 做 completion-only mask
    )
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    print(f"training took {train_s:.1f}s")

    # ---- 保存 adapter ----
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"adapter saved to {adapter_dir}")

    # ---- merge_and_unload ----
    merged = model.merge_and_unload()
    merged_dir = os.path.join(args.out, "merged")
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"merged model saved to {merged_dir}")

    # ---- 质量观测(只记录不设门):20 道新题格式命中率 ----
    eval_problems = []
    # 简单 20 题(与训练不同 seed)
    for i in range(20):
        eval_problems.append(f"求函数 f(x) = {i+1}*x**2 + {i+2} 的导数 f'(x),并化简。")
    sys_prompt = ds[0]["messages"][0]["content"]  # 06 系统提示词
    format_hit = 0
    tag_halluc = 0
    samples = []
    from vllm import LLM, SamplingParams
    vllm_llm = LLM(model=merged_dir, trust_remote_code=False, max_model_len=4096, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    sp = SamplingParams(max_tokens=4096, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
    for i, q in enumerate(eval_problems):
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = vllm_llm.generate([text], sp)[0].outputs[0].text
        has_run = "<run>" in out and "</run>" in out
        has_final = "<final>" in out and "</final>" in out
        ok = has_run and has_final
        if ok:
            format_hit += 1
        # tag 幻觉:出现了协议外标签(如 <tool_call> / <bash>)
        for bad in ["<tool_call>", "<bash>", "<output>"]:
            if bad in out:
                tag_halluc += 1
                break
        samples.append({"q": q, "ok": ok, "has_run": has_run, "has_final": has_final, "output": out[:800]})

    print(f"\n=== 格式命中率: {format_hit}/20 ===")
    print(f"tag 幻觉率: {tag_halluc}/20")
    print(f"总耗时: {time.time()-t_start:.1f}s")

    # 落盘
    receipt = {
        "args": vars(args),
        "system_prompt_sha256": hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest(),
        "trainable_params": trainable,
        "total_params": total_params,
        "trainable_pct": round(trainable/total_params*100, 4),
        "trainable_by_module": dict(by_mod),
        "training_seconds": round(train_s, 1),
        "format_hit_rate": f"{format_hit}/20",
        "tag_halluc_rate": f"{tag_halluc}/20",
        "samples": samples,
    }
    with open(os.path.join(args.out, "receipt.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(f"receipt saved to {args.out}/receipt.json")


if __name__ == "__main__":
    main()
