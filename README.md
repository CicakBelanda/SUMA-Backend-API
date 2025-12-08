# Backend API

FastAPI service for summarizing news articles and YouTube videos using a BART summarizer and Whisper transcription.

## Requirements
- Python 3.10
- FFmpeg installed and available on PATH (required by yt_dlp and Whisper)

## Local Development
```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # on Windows
# source .venv/bin/activate  # on macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API
- `POST /summarize-news` with JSON `{"url": "<cnn/nbc/bbc link>"}`
- `POST /summarize-video` with JSON `{"url": "<youtube link>"}`

## Deployment on Railway
1. Push the `backend` folder to your repository.
2. Create a new Railway project and select the repo.
3. Set the root to `backend/` if prompted.
4. Railway detects the `Procfile` and installs `requirements.txt`.
5. Deploy; the service listens on the provided `$PORT`.
