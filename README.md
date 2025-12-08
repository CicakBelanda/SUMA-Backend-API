---
title: "News & Video Summarizer API"
emoji: 📰
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# News & Video Summarizer API (FastAPI)

This Space provides an API endpoint for summarizing online news articles
and YouTube videos using a fine-tuned BART model and Whisper.


## Features
- POST `/summarize-news` with `{"url": "<cnn/bbc/nbc link>"}` → JSON summary.
- POST `/summarize-video` with `{"url": "<youtube link>"}` → transcribe (Whisper base) then summarize.
- GET `/` returns basic status; GET `/health` returns healthy.
- CORS open to all origins.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py  # or uvicorn app:app --host 0.0.0.0 --port 7860
