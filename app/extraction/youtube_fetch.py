import html
import json
import re
import time
import xml.etree.ElementTree as ET

import requests
import yt_dlp

# -----------------------------
# Extract Video ID
# -----------------------------
VIDEO_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|live/))([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str:
    match = VIDEO_ID_RE.search(url)

    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL")


# -----------------------------
# Download Caption File
# -----------------------------
def download_caption(url: str) -> str:
    for _ in range(3):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(2)

    raise RuntimeError("Failed to download caption file")


# -----------------------------
# Parse JSON3 captions
# -----------------------------
def parse_json3(raw: str) -> str:
    data = json.loads(raw)

    texts = []

    for event in data.get("events", []):
        for seg in event.get("segs", []):
            text = seg.get("utf8", "").strip()

            if text:
                texts.append(text)

    return " ".join(texts)


# -----------------------------
# Parse XML captions
# -----------------------------
def parse_xml(raw: str) -> str:
    root = ET.fromstring(raw)

    texts = []

    for node in root.iter("text"):
        if node.text:
            texts.append(html.unescape(node.text.strip()))

    return " ".join(texts)


# -----------------------------
# Find English Captions
# -----------------------------
def get_english_caption(captions: dict):

    if not captions:
        return None

    priority = [
        "en",
        "en-US",
        "en-GB",
        "a.en",
    ]

    for lang in priority:
        if lang in captions:
            return captions[lang]

    for key in captions:
        if key.startswith("en"):
            return captions[key]

    return None


# -----------------------------
# Fetch Transcript
# -----------------------------
def fetch_transcript(url: str) -> str:

    video_id = extract_video_id(url)

    ydl_opts = {
        "quiet": True,
        "skip_download": True,

        "writesubtitles": True,
        "writeautomaticsub": True,

        "subtitleslangs": ["en"],

        # Uncomment if YouTube blocks you
        "cookiesfrombrowser": ("chrome",),

        # Optional
        "nocheckcertificate": True,
    }

    try:

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False,
            )

        subtitles = info.get("subtitles", {}) or {}
        automatic = info.get("automatic_captions", {}) or {}

        captions = (
            get_english_caption(subtitles)
            or get_english_caption(automatic)
        )

        if not captions:
            raise RuntimeError(
                "No English subtitles or automatic captions found."
            )

        for cap in captions:

            try:

                raw = download_caption(cap["url"])

                if cap.get("ext") == "json3" or raw.startswith("{"):
                    transcript = parse_json3(raw)
                else:
                    transcript = parse_xml(raw)

                transcript = transcript.strip()

                if transcript:
                    return transcript

            except Exception:
                continue

        raise RuntimeError("Transcript exists but could not be parsed.")

    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}")


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        transcript = fetch_transcript(url)

        print("=" * 80)
        print(transcript[:2000])
        print("=" * 80)

    except Exception as e:
        print(e)