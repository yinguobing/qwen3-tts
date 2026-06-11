# qwen3-tts

Qwen3 TTS API Server — an OpenAI-compatible text-to-speech service powered by the Qwen3 TTS 1.7B custom voice model.

## Features

- **OpenAI-compatible API** — drop-in replacement for the `/v1/audio/speech` endpoint
- **Custom voice synthesis** — generate speech with fine-grained control over voice, temperature, sampling, and more
- **Docker-first** — multi-stage CUDA build with minimal runtime footprint
- **Non-blocking inference** — runs generation in a thread pool so the async event loop stays responsive
- **Configurable via environment variables** — model path, device, timeout, and token limits

## Prerequisites

- NVIDIA GPU with CUDA 12.8 support
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Qwen3 TTS model weights](https://huggingface.co/Qwen/Qwen3-TTS-1.7B-CustomVoice) downloaded to `/data/models/qwen3-tts-1.7b-customvoice` (or set `QWEN3_TTS_MODEL_PATH` to your own path)

## Quick Start

```bash
# Set the model path (if different from the default)
export QWEN3_TTS_MODEL_PATH=/path/to/qwen3-tts-1.7b-customvoice

# Build and run
docker compose up -d
```

The server will start on port **8001**. Check health:

```bash
curl http://localhost:8001/health
```

## API Endpoints

### `GET /health`

Health check (always responds, even while the model is loading).

```json
{
  "status": "ok",
  "device": "cuda:0",
  "model_loaded": true,
  "model": "qwen3-tts-1.7b-customvoice"
}
```

### `POST /v1/audio/speech` — OpenAI-compatible

Generate speech with the same interface as OpenAI's TTS API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | (required) | The text to synthesize |
| `voice` | string | `Vivian` | Speaker name |
| `response_format` | string | `wav` | Output format (`wav` or `mp3`) |
| `temperature` | float | `0.2` | Sampling temperature |
| `top_p` | float | `0.6` | Nucleus sampling threshold |
| `top_k` | int | `15` | Top-K sampling |
| `repetition_penalty` | float | `1.1` | Repetition penalty |
| `seed` | int | auto | Random seed (auto-derived from text+speaker if omitted) |
| `do_sample` | bool | `true` | Whether to use sampling |
| `instruct` | string | `null` | Instruction prompt for the model |
| `subtalker_temperature` | float | `null` | Subtalker temperature override |
| `subtalker_dosample` | bool | `null` | Subtalker do_sample override |
| `subtalker_top_p` | float | `null` | Subtalker top_p override |
| `subtalker_top_k` | int | `null` | Subtalker top_k override |

**Example:**

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，欢迎使用 Qwen3 语音合成服务。",
    "voice": "Vivian",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

With an OpenAI client library:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts-1.7b-customvoice",
    voice="Vivian",
    input="Hello, this is a test.",
) as response:
    response.stream_to_file("output.wav")
```

### `POST /v1/audio/tts` — Simplified

A minimal interface that also supports saving directly to disk on the server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | (required) | The text to synthesize |
| `speaker` | string | `Vivian` | Speaker name |
| `output_path` | string | `null` | Optional server-side output path (returns audio inline if omitted) |

Also accepts the same generation parameters as `/v1/audio/speech` (`temperature`, `top_p`, `top_k`, `seed`, etc.).

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_TTS_MODEL_PATH` | `/data/models/qwen3-tts-1.7b-customvoice` | Path to model weights |
| `QWEN3_TTS_DEVICE` | `cuda:0` | Torch device for inference |
| `QWEN3_TTS_PORT` | `8001` | Host port to expose |
| `QWEN3_TTS_TIMEOUT` | `120` | Inference timeout in seconds |
| `QWEN3_TTS_MAX_TOKENS` | `3072` | Maximum new tokens per generation |

## CLI Options

```bash
python qwen3-tts-server.py --host 0.0.0.0 --port 8001 --device cuda:0 --workers 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8001` | Bind port |
| `--device` | `cuda:0` | Torch device |
| `--workers` | `1` | Uvicorn worker count (inference uses a separate thread pool) |

## Docker Image

A multi-stage build keeps the final image lean:

- **Stage 1 (builder):** CUDA devel image — compiles `flash-attn`, installs PyTorch and `qwen-tts`
- **Stage 2 (runtime):** CUDA runtime image — copies only the installed packages, adds `ffmpeg` and `libsndfile1`

Build standalone without Docker Compose:

```bash
docker build -t qwen3-tts:local .
docker run --gpus all \
  -v /data/models:/data/models:ro \
  -p 8001:8001 \
  qwen3-tts:local
```
