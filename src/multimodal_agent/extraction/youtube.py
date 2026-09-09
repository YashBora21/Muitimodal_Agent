from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yt_dlp

from .audio import transcribe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COOKIE_FILE = Path(
    os.getenv(
        "YOUTUBE_COOKIE_FILE",
        PROJECT_ROOT / "www.youtube.com_cookies.txt",
    )
)
DEFAULT_LANGUAGES = ["en", "en-US", "en-GB"]
YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s<>()\[\]\"']+",
    re.IGNORECASE,
)

# Groq's Whisper endpoint rejects uploads above ~25MB; stay well under that.
MAX_AUDIO_BYTES = 20 * 1024 * 1024
CHUNK_SECONDS = 600  # 10 minutes per chunk when splitting is needed


@dataclass
class YouTubeTranscriptResult:
    transcript: str
    source: str
    language: str = "en"
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)


def _base_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "web_embedded"]
            }
        },
    }

    if COOKIE_FILE.exists():
        opts["cookiefile"] = str(COOKIE_FILE)
        logger.info(f"Using cookies: {COOKIE_FILE}")
    else:
        logger.warning("No cookies file found — may be blocked by YouTube.")

    return opts


def _run(ydl_opts: dict[str, Any], url: str) -> None:
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except yt_dlp.utils.DownloadError:
        if "cookiefile" not in ydl_opts:
            raise

        logger.warning("Cookie session failed; retrying public access.")

        retry_opts = {k: v for k, v in ydl_opts.items() if k != "cookiefile"}

        with yt_dlp.YoutubeDL(retry_opts) as ydl:
            ydl.download([url])


def _captions_text(sub_path: str) -> str:
    """Strip VTT timing/markup down to plain text, dropping repeated cue lines."""
    raw = Path(sub_path).read_text(encoding="utf-8", errors="ignore")

    lines: list[str] = []

    for line in raw.splitlines():
        line = line.strip()

        if (
            not line
            or line.isdigit()
            or line.upper().startswith("WEBVTT")
            or "-->" in line
        ):
            continue

        line = re.sub(r"<[^>]+>", "", line)

        if line and (not lines or line != lines[-1]):
            lines.append(line)

    return " ".join(lines)


def _fetch_captions(url: str) -> YouTubeTranscriptResult | None:
    """Try YouTube's own caption track first: it's plain text, so size is never an issue."""
    with tempfile.TemporaryDirectory() as temp_dir:
        opts = _base_opts()

        opts.update(
            {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": DEFAULT_LANGUAGES,
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(temp_dir, "captions.%(ext)s"),
            }
        )

        try:
            _run(opts, url)
        except Exception as e:
            logger.info(f"No captions available, will fall back to audio: {e}")
            return None

        vtt_files = sorted(Path(temp_dir).glob("captions*.vtt"))

        if not vtt_files:
            return None

        text = _captions_text(str(vtt_files[0]))

        if not text.strip():
            return None

        lang = vtt_files[0].stem.split(".")[-1] or "en"

        return YouTubeTranscriptResult(
            transcript=text,
            source="captions",
            language=lang,
        )


def _split_audio(audio_path: str, temp_dir: str) -> list[str]:
    """Split into fixed-length chunks if the file is too big for a single upload."""
    if os.path.getsize(audio_path) <= MAX_AUDIO_BYTES:
        return [audio_path]

    pattern = os.path.join(temp_dir, "chunk_%03d.mp3")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-f",
            "segment",
            "-segment_time",
            str(CHUNK_SECONDS),
            "-c",
            "copy",
            pattern,
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )

    chunks = sorted(Path(temp_dir).glob("chunk_*.mp3"))

    if not chunks:
        raise RuntimeError("Audio splitting produced no chunks.")

    return [str(c) for c in chunks]


def _fetch_via_whisper(url: str) -> YouTubeTranscriptResult:
    with tempfile.TemporaryDirectory() as temp_dir:
        opts = _base_opts()

        opts["format"] = "bestaudio/best"
        opts["outtmpl"] = os.path.join(temp_dir, "audio.%(ext)s")
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ]
        opts["postprocessor_args"] = {
            "FFmpegExtractAudio": ["-ac", "1"]
        }  # mono, smaller files

        try:
            _run(opts, url)

            audio_path = os.path.join(temp_dir, "audio.mp3")

            if not os.path.exists(audio_path):
                candidates = [
                    os.path.join(temp_dir, f)
                    for f in os.listdir(temp_dir)
                    if f.endswith((".mp3", ".m4a", ".webm", ".opus", ".wav"))
                ]

                if not candidates:
                    raise RuntimeError("yt-dlp completed but no audio file found.")

                audio_path = candidates[0]

            logger.info(f"Audio downloaded: {audio_path}")

            chunks = _split_audio(audio_path, temp_dir)

            texts: list[str] = []
            total_duration = 0.0

            for chunk_path in chunks:
                with open(chunk_path, "rb") as f:
                    chunk_data = f.read()

                text, duration = transcribe(chunk_data, os.path.basename(chunk_path))

                if text.strip():
                    texts.append(text.strip())

                total_duration += duration or 0.0

            transcript = " ".join(texts)

            if not transcript:
                raise RuntimeError("Whisper returned empty transcript.")

            return YouTubeTranscriptResult(
                transcript=transcript,
                source="whisper" if len(chunks) == 1 else "whisper-chunked",
                duration=total_duration,
            )

        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(f"yt-dlp download failed: {e}")

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {type(e).__name__}: {e}")


def fetch_transcript(url: str) -> YouTubeTranscriptResult:
    """Prefer YouTube's own captions; fall back to downloading audio and running Whisper."""
    captions = _fetch_captions(url)

    if captions:
        return captions

    return _fetch_via_whisper(url)


def youtube_id(url: str) -> str | None:
    parsed = urlparse(url.rstrip(".,;!?"))
    host = parsed.netloc.lower()

    if host.endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0] or None
    if "youtube.com" not in host:
        return None
    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]

    parts = parsed.path.strip("/").split("/")
    if len(parts) > 1 and parts[0] in {"shorts", "embed", "live"}:
        return parts[1]
    return None


def youtube_urls(text: str) -> list[str]:
    urls = (url.rstrip(".,;!?") for url in YOUTUBE_RE.findall(text))
    return list(dict.fromkeys(url for url in urls if youtube_id(url)))


@cache
def youtube_text(url: str) -> str:
    if not youtube_id(url):
        return "Transcript unavailable: invalid YouTube URL."
    try:
        result = fetch_transcript(url)
        logger.info(
            "Transcript source=%s length=%s",
            result.source,
            len(result.transcript),
        )
        return result.transcript
    except Exception as error:
        logger.error("YouTube transcript failed for %s: %s", url, error)
        return f"Transcript unavailable: {type(error).__name__}: {error}"
