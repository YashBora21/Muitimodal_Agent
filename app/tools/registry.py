import time
from collections.abc import Callable

from app.llm.factory import get_llm_client
from app.state import AgentState


def _timed(name: str, fn: Callable[[AgentState], str]):
    """Wrap a tool fn so every call produces a trace entry, success or failure."""
    def run(state: AgentState) -> None:
        start = time.monotonic()
        try:
            message = fn(state)
            state.log(name, "ok", message, (time.monotonic() - start) * 1000)
        except Exception as e:
            state.warnings.append(f"{name} failed: {e}")
            state.log(name, "failed", str(e), (time.monotonic() - start) * 1000)
    return run


from app.extraction.audio_transcribe import transcribe_audio
from app.extraction.image_ocr import extract_image_text
from app.extraction.pdf_extract import extract_pdf_text
from app.extraction.youtube_fetch import fetch_transcript

# ---- extraction tools ----

def _pdf_extract(state: AgentState) -> str:
    path = state.file_paths.get("pdf")
    if not path:
        raise ValueError("no PDF was attached")
    text, urls = extract_pdf_text(path)
    state.pdf_text = text
    for url in urls:
        if url not in state.youtube_urls:
            state.youtube_urls.append(url)
    return f"extracted {len(text)} chars" + (f", found {len(urls)} YouTube URL(s)" if urls else "")


def _image_ocr(state: AgentState) -> str:
    path = state.file_paths.get("image")
    if not path:
        raise ValueError("no image was attached")
    text, confidence = extract_image_text(path)
    state.image_text = text
    return f"extracted {len(text)} chars, OCR confidence {confidence}%"


def _audio_transcribe(state: AgentState) -> str:
    path = state.file_paths.get("audio")
    if not path:
        raise ValueError("no audio file was attached")
    text, duration = transcribe_audio(path)
    state.audio_text = text
    return f"transcribed {duration:.1f}s of audio"


def _youtube_transcript(state: AgentState) -> str:
    if not state.youtube_urls:
        raise ValueError("no YouTube URL was found to fetch")

    result = youtube_tool(state.youtube_urls[0])

    state.youtube_transcript = result["transcript"]
    state.youtube_metadata = result["metadata"]

    return (
        f"Fetched transcript using {result['source']} "
        f"({len(result['transcript'])} chars)"
    )


# ---- LLM-only tools: fully working right now ----

def _summarize(state: AgentState) -> str:
    text = state.combined_text() or state.query
    client = get_llm_client()
    reply = client.complete(
        system=(
            "Summarize the given text. Respond with exactly three labeled sections:\n"
            "ONE-LINE: <one sentence>\nBULLETS:\n- <bullet>\n- <bullet>\n- <bullet>\n"
            "FIVE-SENTENCE: <five sentence summary>"
        ),
        user=text,
    )
    state.result = {"type": "summary", "content": reply}
    return "summary generated"


def _sentiment(state: AgentState) -> str:
    text = state.combined_text() or state.query
    client = get_llm_client()
    reply = client.complete(
        system=(
            "Classify the sentiment of the given text. Respond with exactly:\n"
            "LABEL: <positive|negative|neutral|mixed>\nCONFIDENCE: <0-1>\nJUSTIFICATION: <one line>"
        ),
        user=text,
    )
    state.result = {"type": "sentiment", "content": reply}
    return "sentiment analyzed"


def _code_explain(state: AgentState) -> str:
    text = state.combined_text() or state.query
    # Don't raise here — if OCR was empty, fall back to query text
    # Raising causes status="failed" so "code_explain" never appears as ok in trace
    client = get_llm_client()
    reply = client.complete(
        system=(
            "Explain the given code. Respond with exactly:\n"
            "LANGUAGE: <detected language>\nEXPLANATION: <what it does>\n"
            "BUGS: <any bugs found, or 'none found'>\nTIME_COMPLEXITY: <Big-O>"
        ),
        user=text or "No code provided.",
    )
    state.result = {"type": "code_explanation", "content": reply}
    return "code explained"


def _compare(state: AgentState) -> str:
    client = get_llm_client()
    reply = client.complete(
        system="""
    You compare multiple extracted sources.

    Use all supplied sources.

    Highlight agreements, differences and answer the user's question.

    Do not summarize independently.
    """,
        user=f"""
    User Question:
    {state.query}

    Sources:
    {state.combined_text()}
    """,
    )
    state.result = {"type": "comparison", "content": reply}
    return "comparison generated"


def _conversational(state: AgentState) -> str:
    client = get_llm_client()
    reply = client.complete(
        system="You are a friendly, helpful assistant. Answer directly and concisely.",
        user=state.query,
    )
    state.result = {"type": "conversation", "content": reply}
    return "answered"


TOOLS: dict[str, Callable[[AgentState], None]] = {
    "pdf_extract": _timed("pdf_extract", _pdf_extract),
    "image_ocr": _timed("image_ocr", _image_ocr),
    "audio_transcribe": _timed("audio_transcribe", _audio_transcribe),
    "youtube_transcript": _timed("youtube_transcript", _youtube_transcript),
    "summarize": _timed("summarize", _summarize),
    "sentiment": _timed("sentiment", _sentiment),
    "code_explain": _timed("code_explain", _code_explain),
    "compare": _timed("compare", _compare),
    "conversational": _timed("conversational", _conversational),
}
