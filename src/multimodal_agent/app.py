from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import Request
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database
from .auth import create_access_token, decode_access_token, hash_password, verify_password
from .graph import graph, stateless_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

PACKAGE_DIR = Path(__file__).resolve().parent
HTML = (PACKAGE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
MAX_BYTES = int(os.getenv("MAX_FILE_MB", "25")) * 1024 * 1024

logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(database.init_db)
    try:
        yield
    finally:
        await asyncio.to_thread(database.close_pool)


app = FastAPI(title="Context-aware Multimodal Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    logger.info(
        "%s %s -> %s (%.4fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response
class Credentials(BaseModel):
    email: str
    password: str
    display_name: str | None = None


class ThreadCreate(BaseModel):
    title: str = "New Conversation"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health", tags=["system"])
def health():
    if not database.enabled():
        raise HTTPException(503, "Database is not configured.")
    try:
        with database.connect() as connection:
            connection.execute("SELECT 1")
    except Exception:
        logger.exception("Database health check failed")
        raise HTTPException(503, "Database is unavailable.")
    return {"status": "healthy", "database": "connected"}


def current_user(authorization: str | None = Header(None)) -> dict[str, str] | None:
    if not database.enabled():
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required.")
    try:
        payload = decode_access_token(authorization.removeprefix("Bearer "))
        if payload.get("type") != "access":
            raise ValueError("Invalid token type")
        user = database.user_by_id(payload["sub"])
    except Exception as error:
        raise HTTPException(401, "Invalid or expired access token.") from error
    if not user:
        raise HTTPException(401, "User no longer exists.")
    return user


def login_response(user: dict[str, str]) -> dict:
    public_user = {"id": user["id"], "email": user["email"], "display_name": user.get("display_name")}
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "user": public_user}


async def read_uploads(files: list[UploadFile]) -> list[dict]:
    uploads = []
    for file in files:
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{file.filename} exceeds the upload limit.")
        uploads.append({"name": file.filename or "upload", "mime": file.content_type, "data": data})
    return uploads


@app.post("/auth/register")
def register(credentials: Credentials):
    if not database.enabled():
        raise HTTPException(503, "Set DATABASE_URL to enable accounts.")
    if "@" not in credentials.email or len(credentials.password) < 8:
        raise HTTPException(400, "Use a valid email and a password of at least 8 characters.")
    try:
        user = database.create_user(credentials.email, hash_password(credentials.password), credentials.display_name)
    except Exception as error:
        raise error
    return login_response(user)


@app.post("/auth/login")
def login(credentials: Credentials):
    if not database.enabled():
        raise HTTPException(503, "Set DATABASE_URL to enable accounts.")
    user = database.user_by_email(credentials.email)
    if not user or not verify_password(user["password_hash"], credentials.password):
        raise HTTPException(401, "Invalid email or password.")
    return login_response(user)


@app.get("/auth/me")
def me(user: dict[str, str] | None = Depends(current_user)):
    return {"database_enabled": database.enabled(), "user": user}


@app.post("/threads", status_code=201)
def create_thread(payload: ThreadCreate, user: dict[str, str] | None = Depends(current_user)):
    if not database.enabled():
        raise HTTPException(503, "Set DATABASE_URL to enable persistent threads.")
    return database.create_thread(user["id"], payload.title)


@app.get("/threads")
def list_threads(user: dict[str, str] | None = Depends(current_user)):
    if not database.enabled():
        return []
    return database.list_threads(user["id"])


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str, user: dict[str, str] | None = Depends(current_user)):
    if not database.enabled():
        raise HTTPException(503, "Set DATABASE_URL to enable persistent threads.")
    thread = database.thread_detail(thread_id, user["id"])
    if not thread:
        raise HTTPException(404, "Conversation not found.")
    return thread


@app.delete("/threads/{thread_id}", status_code=204)
def delete_thread(thread_id: str, user: dict[str, str] | None = Depends(current_user)):
    if not database.enabled():
        raise HTTPException(503, "Set DATABASE_URL to enable persistent threads.")
    if not database.delete_thread(thread_id, user["id"]):
        raise HTTPException(404, "Conversation not found.")


@app.post("/chat")
async def chat(
    message: str = Form(""),
    thread_id: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    user: dict[str, str] | None = Depends(current_user),
):
    if not message.strip() and not files:
        raise HTTPException(400, "Send a message or at least one file.")

    uploads = await read_uploads(files)

    selected_thread_id = thread_id or str(uuid.uuid4())
    try:
        uuid.UUID(selected_thread_id)
    except ValueError as error:
        raise HTTPException(400, "thread_id must be a UUID.") from error

    try:
        if database.enabled():
            await asyncio.to_thread(
                database.ensure_thread, selected_thread_id, user["id"], message.strip() or "New Conversation"
            )
            assets, history = await asyncio.to_thread(database.load_thread, selected_thread_id, user["id"])
            result = await asyncio.to_thread(
                stateless_graph.invoke,
                {"request": message, "uploads": uploads, "assets": assets, "messages": history},
            )
            await asyncio.to_thread(database.save_assets, selected_thread_id, result.get("assets", []))
            await asyncio.to_thread(database.save_messages, selected_thread_id, message, result.get("answer", ""))
        else:
            result = await asyncio.to_thread(
                graph.invoke,
                {"request": message, "uploads": uploads},
                {"configurable": {"thread_id": selected_thread_id}},
            )
        return {
            **{key: result.get(key) for key in ("answer", "decision", "extracted", "logs")},
            "thread_id": selected_thread_id,
            "persistent": database.enabled(),
        }
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(502, f"Processing failed: {type(error).__name__}: {error}") from error
