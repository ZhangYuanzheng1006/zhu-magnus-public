"""Short 27B teacher generation smoke for one data-parallel worker."""
from __future__ import annotations
import json, os, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("TEACHER_MODEL", "/data/magnus/models/Qwen3.8-27B-20260828")
DATA = os.environ.get("TEACHER_DATA", "/data/magnus/closedloop-0828/p1/problems.jsonl")
OUT = Path(os.environ.get("TEACHER_OUT", "/data/magnus/closedloop-0828/teacher-dp-smoke"))
RANK = int(os.environ.get("TEACHER_RANK", "0"))
N = int(os.environ.get("TEACHER_PROMPTS", "3"))
MAX_NEW = int(os.environ.get("TEACHER_MAX_NEW", "256"))

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qs=[]
    with Path(DATA).open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            if r.get("split") in {"dev","eval","test","holdout_family"}:
                qs.append(str(r.get("prompt",r.get("question",""))))
                if len(qs)>=N: break
    t0=time.time(); tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,device_map="cuda",trust_remote_code=False)
    model.eval(); load_s=time.time()-t0
    texts=[tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True,chat_template_kwargs={"enable_thinking":True,"reasoning_effort":"medium"}) for q in qs]
    inputs=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=8192).to(model.device)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t1=time.time()
    with torch.no_grad(): out=model.generate(**inputs,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.pad_token_id)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    gen_s=time.time()-t1; width=inputs["input_ids"].shape[1]; new_tokens=int((out[:,width:]!=tok.pad_token_id).sum().item())
    rec={"rank":RANK,"model":MODEL,"prompts":len(qs),"max_new_tokens":MAX_NEW,"load_s":round(load_s,2),"generate_s":round(gen_s,2),"new_tokens":new_tokens,"tokens_per_s":round(new_tokens/max(gen_s,1e-6),3),"peak_vram_gib":round(torch.cuda.max_memory_allocated()/2**30,3) if torch.cuda.is_available() else None,"torch":torch.__version__}
    (OUT/f"rank-{RANK}.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
    print("TEACHER_DP_SMOKE",json.dumps(rec,ensure_ascii=False),flush=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
