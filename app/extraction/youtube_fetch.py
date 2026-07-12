import re

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

try:
    from youtube_transcript_api._errors import IpBlocked, RequestBlocked
except ImportError:
    class IpBlocked(Exception):
        pass

    class RequestBlocked(Exception):
        pass

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
            segments = YouTubeTranscriptApi.get_transcript(video_id)
            texts = [seg["text"] for seg in segments]
        else:
            fetched = YouTubeTranscriptApi().fetch(video_id)
            texts = [snippet.text for snippet in fetched]
    except TranscriptsDisabled:
        raise RuntimeError("transcript is disabled for this video")
    except NoTranscriptFound:
        raise RuntimeError("no transcript available for this video")
    except VideoUnavailable:
        raise RuntimeError("video is unavailable or private")
    except (IpBlocked, RequestBlocked):
        raise RuntimeError(
            "YouTube is temporarily rate-limiting transcript requests from this network - "
            "try again in a few minutes, or with a different video"
        )
    except Exception as e:
        raise RuntimeError(f"could not fetch transcript: {e}")

    return " ".join(texts)