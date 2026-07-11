import re
import json
import requests
import yt_dlp

VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/)([\w-]{11})")


def extract_video_id(url: str) -> str | None:
    match = VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def fetch_transcript(url: str) -> str:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"could not parse a video id out of: {url}")

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False
            )

            subs = info.get("subtitles", {}) or {}
            auto = info.get("automatic_captions", {}) or {}
            captions = subs.get("en") or auto.get("en") or []

            if not captions:
                raise RuntimeError("no English transcript available")

            # Try each format until one works
            transcript = ""
            for cap in captions:
                try:
                    resp = requests.get(cap["url"], timeout=10)
                    resp.raise_for_status()
                    raw = resp.text.strip()

                    if cap.get("ext") == "json3" or raw.startswith("{"):
                        # JSON format
                        data = json.loads(raw)
                        events = data.get("events", [])
                        texts = []
                        for event in events:
                            for seg in event.get("segs", []):
                                t = seg.get("utf8", "").strip()
                                if t and t != "\n":
                                    texts.append(t)
                        transcript = " ".join(texts)
                    else:
                        # XML format
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(raw)
                        transcript = " ".join(
                            (el.text or "").strip()
                            for el in root.iter("text")
                            if el.text
                        )

                    if transcript.strip():
                        break
                except Exception:
                    continue

            if not transcript.strip():
                raise RuntimeError("transcript was empty after parsing")

            return transcript.strip()

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}")