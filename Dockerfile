FROM python:3.10-slim

WORKDIR /app

# System deps for readability/yt_dlp/whisper
RUN apt-get update && \
    apt-get install -y ffmpeg libxml2 libxslt1.1 libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
