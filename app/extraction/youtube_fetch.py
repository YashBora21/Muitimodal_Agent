import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel


def get_video_metadata(url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "id": info["id"],
        "title": info["title"],
        "channel": info["uploader"],
        "duration": info["duration"],
    }


def transcript_api(video_id):

    transcript = YouTubeTranscriptApi.fetch(video_id)

    return " ".join(chunk.text for chunk in transcript)


def whisper_transcript(url):

    os.makedirs("temp", exist_ok=True)

    output = "temp/audio"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio = None

    for f in os.listdir("temp"):
        if f.startswith("audio"):
            audio = os.path.join("temp", f)
            break

    if audio is None:
        raise RuntimeError("Audio download failed")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    segments, _ = model.transcribe(audio)

    return " ".join(segment.text for segment in segments)


def youtube_tool(url):

    metadata = get_video_metadata(url)

    try:

        transcript = transcript_api(metadata["id"])

        source = "youtube transcript"

    except Exception:

        transcript = whisper_transcript(url)

        source = "whisper"

    return {
        "metadata": metadata,
        "transcript": transcript,
        "source": source
    }