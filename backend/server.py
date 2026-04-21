from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import io
import base64
import logging
import uuid
import time
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from PIL import Image

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Force HF caches to writable dir
os.environ.setdefault("HF_HOME", "/app/backend/.hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/backend/.hf_cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("watermark")

app = FastAPI(title="LLM Watermark Studio")
api_router = APIRouter(prefix="/api")

# In-memory session store for generated runs (process-local; fine for single-pod demo)
SESSIONS: Dict[str, Dict[str, Any]] = {}

SUPPORTED_MODELS = ["gpt2", "gpt2-medium", "facebook/opt-125m"]
BUILTIN_PATTERNS = ["square", "checkerboard", "circle", "diamond", "cross"]


# ---------------- Pydantic models ---------------- #

class GenerateRequest(BaseModel):
    prompt: str
    model_name: str = Field(default="gpt2")
    max_new_tokens: int = Field(default=120, ge=20, le=300)
    pattern: str = Field(default="square")  # built-in choice OR "upload"
    pattern_image_b64: Optional[str] = None  # base64-encoded image if pattern == "upload"
    secret_key: str = Field(default="llmwatermark")
    gamma: float = Field(default=0.5, ge=0.1, le=0.9)
    delta: float = Field(default=0.3, ge=0.05, le=2.0)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_p: float = Field(default=0.9, ge=0.5, le=1.0)
    tau: float = Field(default=0.75, ge=0.5, le=1.0)


class DetectRequest(BaseModel):
    session_id: str
    secret_key: Optional[str] = None
    gamma: Optional[float] = None
    tau: Optional[float] = None


class InfoResponse(BaseModel):
    models: List[str]
    patterns: List[str]


# ---------------- Helpers ---------------- #

def _decode_b64_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _build_watermark_image(pattern: str, pattern_image_b64: Optional[str]) -> Image.Image:
    from image_utils import make_builtin_pattern
    if pattern == "upload":
        if not pattern_image_b64:
            raise HTTPException(status_code=400, detail="pattern_image_b64 is required when pattern='upload'")
        return _decode_b64_image(pattern_image_b64)
    if pattern not in BUILTIN_PATTERNS:
        raise HTTPException(status_code=400, detail=f"Unknown pattern '{pattern}'")
    return make_builtin_pattern(pattern)


# ---------------- Routes ---------------- #

@api_router.get("/")
async def root():
    return {"message": "LLM Watermark Studio API", "ok": True}


@api_router.get("/watermark/info", response_model=InfoResponse)
async def watermark_info():
    return InfoResponse(models=SUPPORTED_MODELS, patterns=BUILTIN_PATTERNS)


@api_router.post("/watermark/generate")
async def watermark_generate(req: GenerateRequest):
    if req.model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"model_name must be one of {SUPPORTED_MODELS}")

    wm_image = _build_watermark_image(req.pattern, req.pattern_image_b64)

    def _run():
        from llm_engine import generate_watermarked
        return generate_watermarked(
            prompt=req.prompt,
            watermark_image=wm_image,
            model_name=req.model_name,
            max_new_tokens=req.max_new_tokens,
            secret_key=req.secret_key,
            gamma=req.gamma,
            delta=req.delta,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    t0 = time.time()
    try:
        result = await run_in_threadpool(_run)
    except Exception as e:
        logger.exception("generate_watermarked failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    elapsed = round(time.time() - t0, 2)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "generated_ids": result["generated_ids"],
        "last_prompt_token": result["last_prompt_token"],
        "pattern_bits": result["pattern_bits"],
        "rows": result["rows"],
        "cols": result["cols"],
        "model_name": result["model_name"],
        "secret_key": result["secret_key"],
        "gamma": result["gamma"],
        "delta": result["delta"],
        "tau": req.tau,
    }

    return {
        "session_id": session_id,
        "generated_text": result["generated_text"],
        "token_count": len(result["generated_ids"]),
        "rows": result["rows"],
        "cols": result["cols"],
        "pattern_length": len(result["pattern_bits"]),
        "model_name": result["model_name"],
        "delta": result["delta"],
        "gamma": result["gamma"],
        "tau": req.tau,
        "target_grid": result["target_grid"],
        "elapsed_s": elapsed,
    }


@api_router.post("/watermark/detect")
async def watermark_detect(req: DetectRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Run generate first.")

    secret_key = req.secret_key or session["secret_key"]
    gamma = req.gamma if req.gamma is not None else session["gamma"]
    tau = req.tau if req.tau is not None else session["tau"]

    def _run():
        from llm_engine import detect_watermark
        return detect_watermark(
            model_name=session["model_name"],
            secret_key=secret_key,
            gamma=gamma,
            tau=tau,
            generated_ids=session["generated_ids"],
            last_prompt_token=session["last_prompt_token"],
            rows=session["rows"],
            cols=session["cols"],
            pattern_bits=session["pattern_bits"],
        )

    try:
        det = await run_in_threadpool(_run)
    except Exception as e:
        logger.exception("detect_watermark failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    if "error" in det:
        raise HTTPException(status_code=400, detail=det["error"])

    det["tau"] = tau
    return det


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
