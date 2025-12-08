import logging
import os
import re
import tempfile
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document
from newspaper import Article


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


def is_valid_news_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in ALLOWED_DOMAINS
    except Exception:
        logging.exception("URL validation failed for %s", url)
        return False


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(" ", strip=True)
    return clean_text(text)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_article_content(url: str) -> str:
    article_text = ""
    try:
        article = Article(url)
        article.download()
        article.parse()
        article_text = clean_text(article.text)
        logging.info("Article scraped via newspaper3k")
    except Exception:
        logging.exception("Primary article scrape failed, falling back to readability/BeautifulSoup")

    if article_text:
        return article_text

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
    except Exception:
        logging.exception("Fallback scraping failed")

    return article_text


def chunk_text(text: str, max_words: int = 800) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def summarize_text(text: str, summarizer) -> str:
    chunks = chunk_text(text)
    if not chunks:
        return ""

    partial_summaries: List[str] = []
    for chunk in chunks:
        try:
            summary = summarizer(
                chunk,
                max_length=300,
                min_length=120,
                do_sample=False,
                truncation=True,
            )[0]["summary_text"]
            partial_summaries.append(clean_text(summary))
        except Exception:
            logging.exception("Summarization failed for chunk")

    merged = clean_text(" ".join(partial_summaries))
    if not merged:
        return ""

    if len(partial_summaries) == 1:
        return merged

    try:
        final_summary = summarizer(
            merged,
            max_length=300,
            min_length=120,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
        return clean_text(final_summary)
    except Exception:
        logging.exception("Final summarization merge failed")
        return merged


def find_first_wav(path: str) -> str:
    if os.path.isfile(path) and path.lower().endswith(".wav"):
        return path
    if os.path.isdir(path):
        for entry in os.listdir(path):
            candidate = os.path.join(path, entry)
            if os.path.isfile(candidate) and candidate.lower().endswith(".wav"):
                return candidate
    return ""


def temp_audio_path() -> str:
    directory = tempfile.mkdtemp(prefix="yt_audio_")
    return os.path.join(directory, "audio.%(ext)s")
