#!/usr/bin/env python3
"""
Qwen3 TTS API Server — OpenAI 兼容接口
修复: 异步不阻塞 + max_new_tokens 兜底 + 多 worker
"""

import argparse
import os
import sys
import time
import io
import asyncio
import concurrent.futures

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse

# ── 全局状态 ──
_model = None
_device = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
_MODEL_PATH = os.environ.get("QWEN3_TTS_MODEL_PATH", "/data/models/qwen3-tts-1.7b-customvoice")
# 每个请求的推理超时（秒），超过强制中断
_INFER_TIMEOUT = int(os.environ.get("QWEN3_TTS_TIMEOUT", "120"))
# max_new_tokens 兜底，防止无限生成
_MAX_NEW_TOKENS = int(os.environ.get("QWEN3_TTS_MAX_TOKENS", "3072"))
# 线程池（单 worker 避免 GPU 争用）
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Qwen3 TTS API")


def get_model():
    """同步加载模型（在线程池中调用）"""
    global _model
    if _model is None:
        print(f"[Qwen3 TTS] Loading model on {_device}...", flush=True)
        t0 = time.time()
        from qwen_tts import Qwen3TTSModel
        _model = Qwen3TTSModel.from_pretrained(
            _MODEL_PATH,
            device_map=_device,
            dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        print(f"[Qwen3 TTS] Model loaded in {time.time()-t0:.1f}s", flush=True)
    return _model


def _do_generate(
    text: str,
    speaker: str,
    # 以下参数由客户端传入，覆盖服务端默认值
    temperature: float = 0.2,
    top_p: float = 0.6,
    top_k: int = 15,
    repetition_penalty: float = 1.1,
    seed: int | None = None,
    do_sample: bool = True,
    instruct: str | None = None,
    # subtalker 子模型参数
    subtalker_temperature: float | None = None,
    subtalker_dosample: bool | None = None,
    subtalker_top_p: float | None = None,
    subtalker_top_k: int | None = None,
):
    """同步生成音频（在线程池中运行）"""
    model = get_model()

    if seed is None:
        seed = hash(text + speaker) & 0x7FFFFFFF

    # 构建 generate_custom_voice 的参数
    gen_kwargs = dict(
        text=text,
        language="chinese",
        speaker=speaker,
        max_new_tokens=_MAX_NEW_TOKENS,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        do_sample=do_sample,
        non_streaming_mode=True,
    )

    if instruct is not None:
        gen_kwargs["instruct"] = instruct
    if subtalker_temperature is not None:
        gen_kwargs["subtalker_temperature"] = subtalker_temperature
    if subtalker_dosample is not None:
        gen_kwargs["subtalker_dosample"] = subtalker_dosample
    if subtalker_top_p is not None:
        gen_kwargs["subtalker_top_p"] = subtalker_top_p
    if subtalker_top_k is not None:
        gen_kwargs["subtalker_top_k"] = subtalker_top_k

    wavs, sr = model.generate_custom_voice(**gen_kwargs)
    return wavs, sr


@app.get("/health")
async def health():
    """健康检查 — 不依赖模型状态，永远即时响应"""
    return {
        "status": "ok",
        "device": _device,
        "model_loaded": _model is not None,
        "model": "qwen3-tts-1.7b-customvoice",
    }


def _extract_tts_params(body: dict) -> dict:
    """从请求 body 提取 TTS 生成参数"""
    params = {}
    for key in ("temperature", "top_p", "top_k", "repetition_penalty",
                 "seed", "do_sample", "instruct",
                 "subtalker_temperature", "subtalker_dosample",
                 "subtalker_top_p", "subtalker_top_k"):
        if key in body:
            params[key] = body[key]
    return params


@app.post("/v1/audio/speech")
async def speech(request: Request):
    """OpenAI 兼容 TTS 接口"""
    body = await request.json()
    text = body.get("input", "")
    voice = body.get("voice", "Vivian")
    response_format = body.get("response_format", "wav")

    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)

    try:
        # 在线程池中执行推理（不阻塞 event loop）
        loop = asyncio.get_event_loop()
        gen_params = _extract_tts_params(body)
        wavs, sr = await asyncio.wait_for(
            loop.run_in_executor(_pool, lambda: _do_generate(text, voice, **gen_params)),
            timeout=_INFER_TIMEOUT,
        )
        wav_data = wavs[0]

        import soundfile as sf

        if response_format == "mp3":
            buf = io.BytesIO()
            sf.write(buf, wav_data, sr, format="mp3")
            media_type = "audio/mpeg"
            content = buf.getvalue()
        else:
            buf = io.BytesIO()
            sf.write(buf, wav_data, sr, format="wav")
            media_type = "audio/wav"
            content = buf.getvalue()

        return Response(content=content, media_type=media_type)

    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": f"推理超时（{_INFER_TIMEOUT}s），请尝试缩短文本"},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/audio/tts")
async def tts_cli(request: Request):
    """简化接口：POST /v1/audio/tts"""
    body = await request.json()
    text = body.get("text", "")
    speaker = body.get("speaker", "Vivian")
    output_path = body.get("output_path", None)

    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    try:
        loop = asyncio.get_event_loop()
        gen_params = _extract_tts_params(body)
        wavs, sr = await asyncio.wait_for(
            loop.run_in_executor(_pool, lambda: _do_generate(text, speaker, **gen_params)),
            timeout=_INFER_TIMEOUT,
        )

        import soundfile as sf

        if output_path:
            sf.write(output_path, wavs[0], sr)
            return {
                "status": "ok",
                "output": output_path,
                "duration": len(wavs[0]) / sr,
            }

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="wav")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": f"推理超时（{_INFER_TIMEOUT}s）"},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 TTS API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1,
                        help="Uvicorn worker 数（默认 1，线程池处理并发）")
    args = parser.parse_args()

    global _device
    _device = args.device

    print(f"[Qwen3 TTS] Starting server on {args.host}:{args.port}")
    print(f"[Qwen3 TTS] Device: {_device}  Workers: {args.workers}")
    print(f"[Qwen3 TTS] Timeout: {_INFER_TIMEOUT}s  MaxTokens: {_MAX_NEW_TOKENS}")
    print(f"[Qwen3 TTS] Model: {_MODEL_PATH}")
    print(f"[Qwen3 TTS] Endpoints:")
    print(f"  GET  /health")
    print(f"  POST /v1/audio/speech  (OpenAI-compatible)")
    print(f"  POST /v1/audio/tts     (simplified)")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
