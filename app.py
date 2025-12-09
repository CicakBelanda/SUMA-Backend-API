import logging
import os
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from readability import Document
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# Optional cache dir to avoid re-downloading models on restarts
os.environ.setdefault("HF_HOME", "/data/hf_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("app")

# Globals for lazy loading
summarizer = None

MODEL_NAME = "brotoo/BART-NewsSummarizer"

ALLOWED_DOMAINS = {
    "cnn.com",
    "www.cnn.com",
    "edition.cnn.com",
    "nbcnews.com",
    "www.nbcnews.com",
    "bbc.com",
    "www.bbc.com",
    "bbc.co.uk",
    "www.bbc.co.uk",
}


class SummarizeNewsRequest(BaseModel):
    url: HttpUrl


class SummarizeVideoRequest(BaseModel):
    url: HttpUrl


def is_valid_news_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in ALLOWED_DOMAINS
    except Exception:
        logger.exception("URL validation failed for %s", url)
        return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return clean_text(soup.get_text(" ", strip=True))


def extract_article_content(url: str) -> str:
    article_text = ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=12, headers=headers)
        response.raise_for_status()
        html = response.text
        document = Document(html)
        article_text = clean_html(document.summary())
        if not article_text:
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            article_text = clean_text(" ".join(paragraphs))
        logger.info("Article scraped with readability/BeautifulSoup")
    except Exception:
        logger.exception("Article scraping failed")
    return article_text


def chunk_text(text: str, max_words: int = 800) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def summarize_text(text: str, model_pipeline) -> str:
    chunks = chunk_text(text)
    if not chunks:
        return ""
    partials: List[str] = []
    for chunk in chunks:
        try:
            summary = model_pipeline(
                chunk,
                max_length=300,
                min_length=120,
                num_beams=4,
                no_repeat_ngram_size=3,
                do_sample=False,
                truncation=True,
            )[0]["summary_text"]
            partials.append(clean_text(summary))
        except Exception:
            logger.exception("Summarization failed for chunk")
    merged = clean_text(" ".join(partials))
    if not merged:
        return ""
    if len(partials) == 1:
        return merged
    try:
        final_summary = model_pipeline(
            merged,
            max_length=300,
            min_length=120,
            num_beams=4,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
        return clean_text(final_summary)
    except Exception:
        logger.exception("Final summarization merge failed")
        return merged


def get_summarizer():
    global summarizer
    if summarizer is None:
        logger.info("Loading summarization model: %s", MODEL_NAME)
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1,  # CPU
        )
        logger.info("Summarization model loaded")
    return summarizer


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0]
    if "youtube.com" in host:
        query_params = parse_qs(parsed.query)
        video_id = query_params.get("v", [""])[0]
        if not video_id and parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[-1].split("/")[0]
        return video_id
    return ""


def extract_youtube_transcript(url: str) -> str:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "id"])
        text = " ".join(segment.get("text", "") for segment in transcript)
        cleaned = clean_text(text)
        if not cleaned:
            raise HTTPException(status_code=500, detail="Transcript was empty.")
        return cleaned
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch YouTube transcript")
        raise HTTPException(status_code=500, detail=f"Could not fetch transcript: {exc}")


app = FastAPI(title="News and Video Summarizer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "message": "API is running"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/summarize-news")
async def summarize_news(payload: SummarizeNewsRequest) -> Dict[str, Any]:
    logger.info("Received news summarization request for %s", payload.url)
    if not is_valid_news_url(str(payload.url)):
        raise HTTPException(status_code=400, detail="Unsupported news domain.")
    try:
        model = get_summarizer()
    except Exception as exc:
        logger.exception("Failed to load summarizer")
        return {"error": f"Model load failed: {exc}"}

    article_text = extract_article_content(str(payload.url))
    if not article_text or len(article_text.split()) < 40:
        raise HTTPException(status_code=400, detail="Could not extract enough article text to summarize.")
    summary = summarize_text(article_text, model)
    if not summary:
        raise HTTPException(status_code=500, detail="Summarization failed.")
    return {"summary": summary}


@app.post("/summarize-video")
async def summarize_video(payload: SummarizeVideoRequest) -> Dict[str, Any]:
    logger.info("Received video summarization request for %s", payload.url)
    if not any(host in str(payload.url) for host in ["youtube.com", "youtu.be"]):
        raise HTTPException(status_code=400, detail="Only YouTube links are supported.")
    try:
        model = get_summarizer()
    except Exception as exc:
        logger.exception("Failed to load summarizer")
        return {"error": f"Model load failed: {exc}"}

    try:
        transcript_text = extract_youtube_transcript(str(payload.url))
        summary = summarize_text(transcript_text, model)
        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed.")
        return {"summary": summary}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during video summarization")
        return {"error": f"Video summarization failed: {exc}"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, workers=1)
