import logging
import os
import re
import tempfile
from typing import Any, Dict, List

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from readability import Document
from transformers import pipeline
import whisper

os.environ.setdefault("HF_HOME", "/data/hf_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("app")

summarizer = None
whisper_model = None

MODEL_NAME = "brotoo/BART-NewsSummarizer"


class SummarizeNewsRequest(BaseModel):
    url: HttpUrl


# === utility clean text ===

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


# === NEWS HANDLER ===

def extract_article_content(url: str) -> str:
    article_text = ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, timeout=12, headers=headers)
        res.raise_for_status()
        html = res.text
        document = Document(html)
        article_text = clean_html(document.summary())
        if not article_text:
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            article_text = clean_text(" ".join(paragraphs))
    except Exception:
        logger.exception("Article scraping failed")
    return article_text


def chunk_text(text: str, max_words: int = 800) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def summarize_text(text: str, model_pipeline) -> str:
    chunks = chunk_text(text)
    partials = []
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
    if len(partials) <= 1:
        return merged

    try:
        final = model_pipeline(
            merged,
            max_length=300,
            min_length=120,
            num_beams=4,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
        return clean_text(final)
    except Exception:
        return merged


def get_summarizer():
    global summarizer
    if summarizer is None:
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1
        )
        logger.info("Summarizer ready")
    return summarizer


# === WHISPER TRANSCRIPTION FOR DIRECT FILE UPLOAD ===

def transcribe_uploaded_video(file_path: str) -> str:
    global whisper_model
    if whisper_model is None:
        model_name = os.getenv("WHISPER_MODEL", "small")
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model(model_name)

    result = whisper_model.transcribe(file_path, fp16=False)
    text = clean_text(result.get("text", ""))
    if not text:
        raise HTTPException(status_code=500, detail="Whisper transcription failed (empty text).")
    return text


# === FASTAPI APP ===

app = FastAPI(title="News and Video Summarizer", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/summarize-upload-video")
async def summarize_upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload video/audio,
    transcribe with Whisper → summarize with BART.
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".m4a", ".wav")):
        raise HTTPException(status_code=400, detail="Only video/audio formats are accepted.")

    tmp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        transcript = transcribe_uploaded_video(temp_path)
        model = get_summarizer()

        summary = summarize_text(transcript, model)
        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed.")
        return {"summary": summary}

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


@app.post("/summarize-news")
async def summarize_news(payload: SummarizeNewsRequest) -> Dict[str, Any]:
    url = str(payload.url)
    logger.info("Received news summarization request for %s", url)

    # ⛔️ DOMAIN CHECK REMOVED — now accepts any domain

    model = get_summarizer()

    article_text = extract_article_content(url)
    if not article_text or len(article_text.split()) < 40:
        raise HTTPException(status_code=400, detail="Could not extract enough article text to summarize.")

    summary = summarize_text(article_text, model)
    if not summary:
        raise HTTPException(status_code=500, detail="Summarization failed.")

    return {"summary": summary}