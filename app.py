import logging
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from readability import Document
from transformers import pipeline
from yt_dlp import YoutubeDL

# Optional cache dir to avoid re-downloading models on restarts
os.environ.setdefault("HF_HOME", "/data/hf_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("app")

# Globals for lazy loading
summarizer = None
whisper_model = None

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


def get_whisper():
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model: base")
        import whisper  # type: ignore

        whisper_model = whisper.load_model("base", device="cpu")
        logger.info("Whisper model loaded")
    return whisper_model


def temp_audio_path() -> str:
    directory = tempfile.mkdtemp(prefix="yt_audio_")
    return os.path.join(directory, "audio.%(ext)s")


def find_first_wav(path: str) -> str:
    if os.path.isfile(path) and path.lower().endswith(".wav"):
        return path
    if os.path.isdir(path):
        for entry in os.listdir(path):
            candidate = os.path.join(path, entry)
            if os.path.isfile(candidate) and candidate.lower().endswith(".wav"):
                return candidate
    return ""


def download_youtube_audio(url: str) -> str:
    output_template = temp_audio_path()
    temp_dir = os.path.dirname(output_template)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    wav_path = find_first_wav(temp_dir)
    if not wav_path:
        raise ValueError("Failed to download or convert YouTube audio.")
    return wav_path


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

    audio_path = ""
    temp_dir = ""
    try:
        whisper = get_whisper()
        audio_path = download_youtube_audio(str(payload.url))
        temp_dir = os.path.dirname(audio_path)
        transcript = whisper.transcribe(audio_path, language="en")
        transcript_text = clean_text(transcript.get("text", ""))
        if not transcript_text:
            raise HTTPException(status_code=500, detail="No transcript text could be produced from the audio.")
        summary = summarize_text(transcript_text, model)
        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed.")
        return {"summary": summary}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during video summarization")
        return {"error": f"Video summarization failed: {exc}"}
    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            logger.exception("Failed to clean up temporary audio files")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, workers=1)
