from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_ITERATIONS = 8
AGENT_MAX_TOKENS = 2000
CHAT_HISTORY_MESSAGES = 12

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 6
RAG_MIN_CHARS = 10_000

WHOLE_DOCUMENT_QUERY = (
    "main topic purpose important points key details "
    "major sections findings outcomes final points"
)

WHOLE_DOCUMENT_TERMS = (
    "summarize",
    "summary",
    "overview",
    "main points",
    "key points",
    "what is this about",
    "what is the document about",
    "entire document",
    "whole document",
)

TESSDATA = Path(
    os.getenv(
        "TESSDATA_PREFIX",
        r"C:\Program Files\Tesseract-OCR\tessdata",
    )
)

YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/"
    r"[^\s<>()\[\]\"']+",
    re.IGNORECASE,
)


def client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )