import logging
from typing import Any, Dict

import torch
import whisper
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from transformers import pipeline

from news import summarize_news_article
from video import summarize_video_url


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "brotoo/BART-NewsSummarizer"
DEVICE_ID = 0 if torch.cuda.is_available() else -1

logger.info("Loading summarization model: %s", MODEL_NAME)
summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE_ID,
)

logger.info("Loading Whisper model: small")
whisper_model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="News and Video Summarizer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeNewsRequest(BaseModel):
    url: HttpUrl


class SummarizeVideoRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize-news")
async def summarize_news(payload: SummarizeNewsRequest) -> Dict[str, Any]:
    logger.info("Received news summarization request")
    try:
        summary = summarize_news_article(str(payload.url), summarizer)
        return {"summary": summary}
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error during news summarization")
        raise HTTPException(status_code=400, detail="Failed to summarize news article.")


@app.post("/summarize-video")
async def summarize_video(payload: SummarizeVideoRequest) -> Dict[str, Any]:
    logger.info("Received video summarization request")
    try:
        summary = summarize_video_url(str(payload.url), summarizer, whisper_model)
        return {"summary": summary}
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error during video summarization")
        raise HTTPException(status_code=400, detail="Failed to summarize video.")
