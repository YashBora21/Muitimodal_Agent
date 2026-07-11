import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.llm.factory import get_llm_client
from app.orchestrator import run_agent
from app.state import AgentState


settings.validate()

app = FastAPI(title="Multimodal Agentic Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    client = get_llm_client()
    return {"status": "ok", "llm_provider": client.name}


class PingBody(BaseModel):
    message: str


@app.post("/ping-llm")
def ping_llm(body: PingBody):
    """
    Sanity check endpoint - not part of the assignment spec, just here so
    you can confirm the Groq/Ollama wiring actually works before we build
    the extraction layer and planner on top of it. Safe to delete later.
    """
    client = get_llm_client()
    reply = client.complete(
        system="You are a terse test assistant. Reply in one short sentence.",
        user=body.message,
    )
    return {"provider": client.name, "reply": reply}


@app.post("/run")
async def run(
    query: str = Form(""),
    pdf: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
):
    """
    Single entrypoint for the whole assignment: text query plus any combination
    of pdf/image/audio in one request. Temp files are cleaned up after the run
    regardless of success or failure.
    """
    state = AgentState(query=query)
    saved_paths: list[str] = []

    for label, upload in (("pdf", pdf), ("image", image), ("audio", audio)):
        if upload is None:
            continue
        suffix = os.path.splitext(upload.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            state.file_paths[label] = tmp.name
            saved_paths.append(tmp.name)
    client = get_llm_client()
    client.reset_usage()
    try:
        state = run_agent(state)
    finally:
        for path in saved_paths:
            os.remove(path)

    return {
        "result": state.result,
        "extracted": {
            "pdf_text": state.pdf_text,
            "image_text": state.image_text,
            "audio_text": state.audio_text,
            "youtube_urls": state.youtube_urls,
            "youtube_transcript": state.youtube_transcript,
        },
        "trace": [step.__dict__ for step in state.trace],
        "warnings": state.warnings,
        "usage": client.get_usage_summary(),   # ← add this

    }


