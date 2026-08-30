"""Native torch DDP timing probe for Qwen3.5-9B (avoids Trainer launcher)."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL=os.environ.get('SPEED_MODEL','/data/magnus/models/Qwen3.5-9B-20260828')
DATA=os.environ.get('SPEED_DATA','/data/magnus/closedloop-0828/p2/sft_trajectories.jsonl')
OUT=Path(os.environ.get('SPEED_OUT','/data/magnus/closedloop-0828/sft-speed/ddp-native'))
STEPS=int(os.environ.get('SPEED_STEPS','3')); MICRO=2; ACCUM=4; MAX_LEN=int(os.environ.get('SPEED_MAX_LEN','2048'))

def main()->int:
 rank=int(os.environ.get('RANK','0')); local=int(os.environ.get('LOCAL_RANK',str(rank))); world=int(os.environ.get('WORLD_SIZE','1'))
 torch.cuda.set_device(local); dist.init_process_group('nccl')
 tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
 rows=[]
 with Path(DATA).open(encoding='utf-8') as f:
  for line in f:
   r=json.loads(line)
   if r.get('protocol_valid') is True:
    rows.append(tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False,chat_template_kwargs={'enable_thinking':False}))
    if len(rows)>=world*MICRO*ACCUM*STEPS:break
 local_rows=rows[rank::world]
 model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False,attn_implementation='sdpa',device_map={'':f'cuda:{local}'})
 model.config.use_cache=False
 model.gradient_checkpointing_enable()
 for n,p in model.named_parameters():
  if any(x in n.lower() for x in ('vision','visual','image_processor','merger')):p.requires_grad=False
 model=get_peft_model(model,LoraConfig(r=32,lora_alpha=32,target_modules='all-linear',lora_dropout=0.05,bias='none',task_type='CAUSAL_LM'))
 model=DDP(model,device_ids=[local],output_device=local,find_unused_parameters=False)
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-5)
 def batch(start):
  enc=[tok(x,truncation=True,max_length=MAX_LEN,add_special_tokens=False)['input_ids'] for x in local_rows[start:start+MICRO]]; w=max(map(len,enc)); pad=tok.pad_token_id or tok.eos_token_id
  ids=torch.full((len(enc),w),pad,dtype=torch.long,device=f'cuda:{local}'); mask=torch.zeros_like(ids)
  for i,s in enumerate(enc): ids[i,:len(s)]=torch.tensor(s,device=ids.device); mask[i,:len(s)]=1
  return ids,mask
 # warmup one optimizer step
 for _ in range(1):
  opt.zero_grad(set_to_none=True)
  for k in range(ACCUM):
   ids,mask=batch(k*MICRO); (model(input_ids=ids,attention_mask=mask,labels=ids).loss/ACCUM).backward()
  opt.step(); dist.barrier()
 if rank==0 and torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
 dist.barrier(); t0=time.perf_counter(); times=[]; grads=[]
 for step in range(STEPS):
  if torch.cuda.is_available():torch.cuda.synchronize()
  st=time.perf_counter(); opt.zero_grad(set_to_none=True)
  for k in range(ACCUM):
   ids,mask=batch((step*ACCUM+k)*MICRO); loss=model(input_ids=ids,attention_mask=mask,labels=ids).loss/ACCUM; loss.backward()
  g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); torch.cuda.synchronize(); times.append(time.perf_counter()-st); grads.append(float(g))
 dist.barrier(); elapsed=sum(times)/len(times); peak=torch.cuda.max_memory_allocated()/2**30
 rec={'rank':rank,'world_size':world,'steps':STEPS,'micro':MICRO,'accum':ACCUM,'global_batch':MICRO*ACCUM*world,'step_s_mean':round(elapsed,3),'step_s_all': [round(x,3) for x in times],'peak_vram_gib':round(peak,3),'grad_norm':grads,'torch':torch.__version__}
 OUT.mkdir(parents=True,exist_ok=True)
 if rank==0:(OUT/'receipt.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
 print('SFT_DDP_NATIVE',json.dumps(rec,ensure_ascii=False),flush=True); dist.destroy_process_group(); return 0
if __name__=='__main__':raise SystemExit(main())
