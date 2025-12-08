import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from news import summarize_news_article
from video import summarize_video_url

# ========== HuggingFace cache directory (optional & safe for Railway) ==========
os.environ["HF_HOME"] = "/app/cache"
os.makedirs("/app/cache", exist_ok=True)

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("app")

# ========== Lazy-loaded model variables ==========
summarizer = None
whisper_model = None

MODEL_NAME = "brotoo/BART-NewsSummarizer"


# ========== Load functions (dijalankan hanya saat dibutuhkan) ==========
def get_summarizer():
    global summarizer
    if summarizer is None:
        logger.info("Loading summarization model: %s", MODEL_NAME)
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1  # CPU only
        )
        logger.info("Summarization model loaded successfully")
    return summarizer


def get_whisper():
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model: base")
        import whisper
        whisper_model = whisper.load_model("base", device="cpu")
        logger.info("Whisper model loaded successfully")
    return whisper_model


# ========== FastAPI App ==========
app = FastAPI(title="News and Video Summarizer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Startup (preload summarizer) ==========
@app.on_event("startup")
async def preload_models() -> None:
    """
    Preload the text summarizer so the first request doesn't wait on model download.
    Failure here should not crash the app; we'll retry lazily on first use.
    """
    try:
        get_summarizer()
    except Exception as exc:
        logger.exception("Preload summarizer failed; will retry on demand: %s", exc)


# ========== Request Schemas ==========
class SummarizeNewsRequest(BaseModel):
    url: HttpUrl


class SummarizeVideoRequest(BaseModel):
    url: HttpUrl


# ========== Health Check ==========
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# ========== API Endpoints ==========
@app.post("/summarize-news")
async def summarize_news(payload: SummarizeNewsRequest) -> Dict[str, Any]:
    logger.info("Received news summarization request")
    try:
        model = get_summarizer()
        summary = summarize_news_article(str(payload.url), model)
        return {"summary": summary}
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during news summarization: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to summarize news article.")


@app.post("/summarize-video")
async def summarize_video(payload: SummarizeVideoRequest) -> Dict[str, Any]:
    logger.info("Received video summarization request")
    try:
        model = get_summarizer()
        whisper = get_whisper()
        summary = summarize_video_url(str(payload.url), model, whisper)
        return {"summary": summary}
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during video summarization: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to summarize video.")
