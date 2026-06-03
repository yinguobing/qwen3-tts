# Qwen3 TTS API Server (CUDA)
# 多阶段构建：编译阶段用 devel 镜像，运行阶段用 runtime 镜像减体积

# ─── Stage 1: Builder（编译 flash-attn） ───
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. 编译依赖（flash-attn 需要 python3-dev + build-essential + git）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 torch + torchaudio（从 cu128 索引，保证 CUDA 版本一致；不需要 torchvision）
RUN pip install --upgrade pip \
    && pip install \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

# 3. 编译 flash-attn
RUN pip install packaging psutil \
    && pip install flash-attn --no-build-isolation --no-cache-dir

# 4. 安装 qwen-tts 及 API 依赖（torchaudio 已预装，pip 不会从 PyPI 拉取不兼容版本）
RUN pip install \
        qwen-tts==0.1.1 \
        fastapi \
        uvicorn[standard] \
        soundfile

# 5. 清理构建临时文件
RUN rm -rf /root/.cache/pip /tmp/*

# ─── Stage 2: Runtime（仅运行时库，无编译工具） ───
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 6. 运行时系统依赖（无需编译工具，体积大幅减小）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# 7. 从 builder 复制全部 Python 包（torch + flash-attn + qwen-tts + 依赖）
COPY --from=builder /usr/local /usr/local

# 8. 复制服务端代码
COPY qwen3-tts-server.py /app/server.py

# 9. 模型缓存
VOLUME /root/.cache

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5).read()" || exit 1

CMD ["python3", "/app/server.py", "--host", "0.0.0.0", "--port", "8001", "--device", "cuda:0"]
