# LFM2.5-8B-A1B 微调/验证环境(预装依赖,权重不打包)
# 基础:ACR 上的 sft-base(基于 pytorch2.5.1+cuda12.4,已装 transformers/peft/bnb)
# 阿里云内网拉取快,避免 Docker Hub 直连慢的问题
FROM crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v5

ENV DEBIAN_FRONTEND=noninteractive \
    HF_ENDPOINT=https://hf-mirror.com \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 升级 transformers 到 >=5.0.0(lfm2_moe 硬要求),装 trl/sympy/scipy/vllm 等
# sft-base 已带旧 transformers/peft/bnb;这里原位升级并补装
RUN pip install --no-cache-dir -U \
    "transformers>=5.0.0" \
    "peft>=0.20.0" \
    "trl>=0.22.0" \
    "datasets" \
    "accelerate" \
    "sentencepiece" \
    "tokenizers" \
    "safetensors" \
    "sympy" \
    "scipy" \
    "huggingface_hub" \
    "modelscope" \
    "einops" \
    "jinja2" \
    "tqdm" \
    "psutil"

# vLLM(CPU/GPU 推理;体积大,单独层便于缓存;失败不阻塞)
RUN pip install --no-cache-dir "vllm>=0.11.0" || echo "[warn] vllm install failed"

# 验证
RUN python -c "import transformers, peft, trl, sympy, scipy; print('transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__)"

WORKDIR /workspace
CMD ["bash"]
