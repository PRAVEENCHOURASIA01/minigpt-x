import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference.generate import generate_stream, sample_next_token
from model.gpt import GPTConfig, MiniGPTX
from tokenizer.encode import BPETokenizer

app_state = {}
generation_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=1000)
    max_new_tokens: int = Field(default=100, ge=1, le=300)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=200)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device_name = os.environ.get("MODEL_DEVICE", None)
    if device_name is None:
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    ckpt_path = os.environ.get("MODEL_CKPT", "checkpoints/smoke_latest.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["config"]["model"])
    model = MiniGPTX(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok_path = ckpt.get("tokenizer_path", "tokenizer/bpe_4096.json")
    if not os.path.exists(tok_path):
        tok_path = "tokenizer/bpe_4096.json"
    tokenizer = BPETokenizer.load(tok_path)

    app_state["model"] = model
    app_state["tokenizer"] = tokenizer
    app_state["config"] = cfg
    app_state["device"] = device
    app_state["ckpt_path"] = ckpt_path
    yield
    app_state.clear()


app = FastAPI(title="MiniGPT-X API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "..", "app", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "MiniGPT-X UI not found"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(app_state.get("device", "cpu")),
        "checkpoint": app_state.get("ckpt_path", ""),
    }


@app.get("/info")
async def info():
    model = app_state.get("model")
    cfg = app_state.get("config")
    device = app_state.get("device")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "device": str(device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "non_embedding_parameters": model.num_parameters(non_embedding=True),
        "config": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "use_rope": cfg.use_rope,
            "norm_type": cfg.norm_type,
            "activation": cfg.activation,
            "use_sdpa": cfg.use_sdpa,
            "tie_embeddings": cfg.tie_embeddings,
        },
    }


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    model = app_state.get("model")
    tokenizer = app_state.get("tokenizer")
    device = app_state.get("device")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if not req.stream:
        async with generation_lock:
            loop = asyncio.get_running_loop()

            def _sync_gen():
                tokens = []
                for chunk in generate_stream(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    use_kv_cache=True,
                    device=device,
                ):
                    tokens.append(chunk)
                return "".join(tokens)

            output_text = await loop.run_in_executor(None, _sync_gen)
            return {"prompt": req.prompt, "text": output_text}

    async def event_generator():
        async with generation_lock:
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()

            def _producer():
                try:
                    for chunk in generate_stream(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=req.prompt,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        use_kv_cache=True,
                        device=device,
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            prod_future = loop.run_in_executor(None, _producer)
            while True:
                item = await q.get()
                if item is None:
                    break
                data = json.dumps({"token": item})
                yield f"data: {data}\n\n"
            await prod_future

    return StreamingResponse(event_generator(), media_type="text/event-stream")
