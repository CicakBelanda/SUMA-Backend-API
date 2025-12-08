import logging
import os
import shutil
import tempfile

from yt_dlp import YoutubeDL

from utils import clean_text, find_first_wav, summarize_text, temp_audio_path


def _download_youtube_audio(url: str) -> str:
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


def summarize_video_url(url: str, summarizer, whisper_model) -> str:
    if not any(host in url for host in ["youtube.com", "youtu.be"]):
        raise ValueError("Only YouTube links are supported.")

    audio_path = ""
    temp_dir = ""
    try:
        audio_path = _download_youtube_audio(url)
        temp_dir = os.path.dirname(audio_path)
        logging.info("Transcribing audio with Whisper")
        transcript = whisper_model.transcribe(audio_path, language="en")
        transcript_text = clean_text(transcript.get("text", ""))
        if not transcript_text:
            raise ValueError("No transcript text could be produced from the audio.")

        logging.info("Generating summary for video transcript")
        summary = summarize_text(transcript_text, summarizer)
        if not summary:
            raise ValueError("Summarization failed for the provided video.")
        return summary
    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            logging.exception("Failed to clean up temporary audio files")
