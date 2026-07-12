import re

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/)([\w-]{11})")


def extract_video_id(url: str) -> str | None:
    match = VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def fetch_transcript(url: str) -> str:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"could not parse a video id out of: {url}")

    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            # library <= 0.6.x - classmethod, list of dicts
            segments = YouTubeTranscriptApi.get_transcript(video_id)
            texts = [seg["text"] for seg in segments]
        else:
            # library >= 1.0 - instance-based, iterable of snippet objects
            fetched = YouTubeTranscriptApi().fetch(video_id)
            texts = [snippet.text for snippet in fetched]
    except TranscriptsDisabled:
        raise RuntimeError("transcript is disabled for this video")
    except NoTranscriptFound:
        raise RuntimeError("no transcript available for this video")
    except VideoUnavailable:
        raise RuntimeError("video is unavailable or private")
    except Exception as e:
        raise RuntimeError(f"could not fetch transcript: {e}")

    return " ".join(texts)