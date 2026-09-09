from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .graph import graph


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

PACKAGE_DIR = Path(__file__).resolve().parent
HTML = (PACKAGE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
MAX_BYTES = int(os.getenv("MAX_FILE_MB", "25")) * 1024 * 1024

app = FastAPI(title="Context-aware Multimodal Agent")
app.mount(
    "/static",
    StaticFiles(directory=PACKAGE_DIR / "static"),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.post("/chat")
async def chat(
    message: str = Form(""),
    thread_id: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    if not message.strip() and not files:
        raise HTTPException(400, "Send a message or at least one file.")

    uploads = []
    for file in files:
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{file.filename} exceeds the upload limit.")
        uploads.append(
            {
                "name": file.filename or "upload",
                "mime": file.content_type,
                "data": data,
            }
        )

    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {"request": message, "uploads": uploads},
            {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}},
        )
        return {
            key: result.get(key)
            for key in ("answer", "decision", "extracted", "logs")
        }
    except (ValueError, RuntimeError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(
            502,
            f"Processing failed: {type(error).__name__}: {error}",
        ) from error
