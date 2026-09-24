from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import Asset
from .retrieval import chunks, needs_hybrid_retrieval

logger = logging.getLogger(__name__)

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL", "")
DIRECT_DATABASE_URL = os.getenv("DATABASE_URL_UNPOOLED", DATABASE_URL)
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "database" / "schema.sql"
PAGE_RE = re.compile(r"(?m)^--- Page (\d+) ---$")
UNSAFE_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_pool = None


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    valid_unicode = value.encode("utf-8", errors="replace").decode("utf-8")
    return UNSAFE_TEXT_RE.sub("", valid_unicode)


def enabled() -> bool:
    return bool(DATABASE_URL)


def connect(url: str | None = None):
    global _pool

    if url:
        import psycopg

        return psycopg.connect(url)

    if _pool is None:
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(DATABASE_URL, min_size=0, max_size=POOL_SIZE)
    return _pool.connection()


def close_pool() -> None:
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None


def init_db() -> None:
    if not enabled():
        return
    with connect(DIRECT_DATABASE_URL) as connection:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def create_user(email: str, password_hash: str, display_name: str | None) -> dict[str, str]:
    email = clean_text(email).lower()
    display_name = clean_text(display_name) or None
    with connect() as connection:
        row = connection.execute(
            """
            INSERT INTO users (email, password_hash, display_name)
            VALUES (%s, %s, %s)
            RETURNING id::text, email, COALESCE(display_name, '')
            """,
            (email, password_hash, display_name),
        ).fetchone()
    return {"id": row[0], "email": row[1], "display_name": row[2]}


def user_by_email(email: str) -> dict[str, str] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id::text, email, password_hash, COALESCE(display_name, '') FROM users WHERE email = %s",
            (clean_text(email).lower(),),
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "email": row[1], "password_hash": row[2], "display_name": row[3]}


def user_by_id(user_id: str) -> dict[str, str] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id::text, email, COALESCE(display_name, '') FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
    return {"id": row[0], "email": row[1], "display_name": row[2]} if row else None


def create_thread(user_id: str, title: str = "New Conversation") -> dict[str, str]:
    title = clean_text(title)
    with connect() as connection:
        row = connection.execute(
            """
            INSERT INTO threads (user_id, title)
            VALUES (%s, %s)
            RETURNING id::text, title, created_at::text, updated_at::text
            """,
            (user_id, title[:255] or "New Conversation"),
        ).fetchone()
    return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}


def list_threads(user_id: str) -> list[dict[str, str]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id::text, title, created_at::text, updated_at::text
            FROM threads WHERE user_id = %s ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}
        for row in rows
    ]


def thread_detail(thread_id: str, user_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        thread = connection.execute(
            """
            SELECT id::text, title, created_at::text, updated_at::text
            FROM threads WHERE id = %s AND user_id = %s
            """,
            (thread_id, user_id),
        ).fetchone()
        if not thread:
            return None
        messages = connection.execute(
            """
            SELECT id::text, role, content, created_at::text
            FROM messages WHERE thread_id = %s ORDER BY created_at, id
            """,
            (thread_id,),
        ).fetchall()
        assets = connection.execute(
            """
            SELECT id::text, asset_key, kind, original_name, processing_status,
                   created_at::text
            FROM assets WHERE thread_id = %s ORDER BY created_at
            """,
            (thread_id,),
        ).fetchall()
    return {
        "id": thread[0],
        "title": thread[1],
        "created_at": thread[2],
        "updated_at": thread[3],
        "messages": [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in messages
        ],
        "assets": [
            {"id": row[0], "asset_key": row[1], "kind": row[2], "name": row[3], "status": row[4], "created_at": row[5]}
            for row in assets
        ],
    }


def delete_thread(thread_id: str, user_id: str) -> bool:
    with connect() as connection:
        result = connection.execute(
            "DELETE FROM threads WHERE id = %s AND user_id = %s",
            (thread_id, user_id),
        )
    return result.rowcount > 0

def ensure_thread(thread_id: str, user_id: str, title: str) -> None:
    title = clean_text(title)
    with connect() as connection:
        owner = connection.execute(
            "SELECT user_id::text, title FROM threads WHERE id = %s", (thread_id,)
        ).fetchone()
        if owner and owner[0] != user_id:
            raise PermissionError("Thread does not belong to this user.")
        if owner and owner[1] == "New Conversation" and title:
            connection.execute(
                "UPDATE threads SET title = %s, updated_at = NOW() WHERE id = %s",
                (title[:255], thread_id),
            )
        if not owner:
            connection.execute(
                "INSERT INTO threads (id, user_id, title) VALUES (%s, %s, %s)",
                (thread_id, user_id, title[:255] or "New Conversation"),
            )


def load_thread(thread_id: str, user_id: str) -> tuple[list[Asset], list[dict[str, str]]]:
    with connect() as connection:
        owner = connection.execute(
            "SELECT 1 FROM threads WHERE id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
        if not owner:
            raise PermissionError("Thread does not belong to this user.")
        asset_rows = connection.execute(
            """
            SELECT id::text, asset_key, kind, original_name, content_text, raw_data
            FROM assets WHERE thread_id = %s ORDER BY created_at
            """,
            (thread_id,),
        ).fetchall()
        message_rows = connection.execute(
            "SELECT role, content FROM messages WHERE thread_id = %s ORDER BY created_at, id",
            (thread_id,),
        ).fetchall()
    assets: list[Asset] = [
        {
            "db_id": row[0], "id": row[1], "kind": row[2], "name": row[3],
            "content": row[4] or "", "data": bytes(row[5]) if row[5] else b"",
        }
        for row in asset_rows
    ]
    logger.info("Database | thread loaded thread=%s assets=%s messages=%s", thread_id, len(assets), len(message_rows))
    return assets, [{"role": row[0], "content": row[1]} for row in message_rows]


def _vectors(parts: list[str]) -> list[list[float] | None]:
    try:
        from .retrieval import _embedder

        return _embedder().encode(parts, normalize_embeddings=True).tolist()
    except Exception:
        return [None] * len(parts)


def save_assets(thread_id: str, assets: list[Asset]) -> None:
    saved_chunks = 0
    with connect() as connection:
        for asset in assets:
            content = clean_text(asset.get("content", ""))
            row = connection.execute(
                """
                INSERT INTO assets (
                    thread_id, asset_key, kind, original_name,
                    content_text, raw_data, file_size_bytes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, asset_key) DO UPDATE SET
                    content_text = EXCLUDED.content_text
                RETURNING id::text
                """,
                (
                    thread_id, clean_text(asset["id"]), clean_text(asset["kind"]), clean_text(asset["name"]),
                    content, asset.get("data"),
                    len(asset.get("data", b"")),
                ),
            ).fetchone()
            asset["db_id"] = row[0]
            if not needs_hybrid_retrieval(content):
                continue
            exists = connection.execute(
                "SELECT 1 FROM asset_chunks WHERE asset_id = %s LIMIT 1", (row[0],)
            ).fetchone()
            if exists:
                continue
            parts = chunks(content)
            for index, (part, vector) in enumerate(zip(parts, _vectors(parts))):
                page_match = list(PAGE_RE.finditer(part))
                page_number = int(page_match[0].group(1)) if page_match else None
                connection.execute(
                    """
                    INSERT INTO asset_chunks (asset_id, chunk_index, content, page_number, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (row[0], index, part, page_number, str(vector) if vector else None),
                )
                saved_chunks += 1
    logger.info("Database | assets saved thread=%s assets=%s chunks=%s", thread_id, len(assets), saved_chunks)


def update_asset_content(asset: Asset) -> None:
    content = clean_text(asset.get("content", ""))
    asset_id = asset.get("db_id")
    if not asset_id:
        return
    saved_chunks = 0
    with connect() as connection:
        connection.execute(
            "UPDATE assets SET content_text = %s WHERE id = %s",
            (content, asset_id),
        )
        connection.execute("DELETE FROM asset_chunks WHERE asset_id = %s", (asset_id,))
        if needs_hybrid_retrieval(content):
            parts = chunks(content)
            for index, (part, vector) in enumerate(zip(parts, _vectors(parts))):
                page_match = PAGE_RE.search(part)
                page_number = int(page_match.group(1)) if page_match else None
                connection.execute(
                    """
                    INSERT INTO asset_chunks (asset_id, chunk_index, content, page_number, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (asset_id, index, part, page_number, str(vector) if vector else None),
                )
                saved_chunks += 1
    logger.info("Database | lazy asset persisted asset=%s chunks=%s", asset_id, saved_chunks)


def save_messages(thread_id: str, request: str, answer: str) -> None:
    request = clean_text(request)
    answer = clean_text(answer)
    with connect() as connection:
        connection.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (%s, 'user', %s)",
            (thread_id, request or "[uploaded files]"),
        )
        connection.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (%s, 'assistant', %s)",
            (thread_id, answer),
        )
        connection.execute("UPDATE threads SET updated_at = NOW() WHERE id = %s", (thread_id,))
    logger.info("Database | messages saved thread=%s user_chars=%s answer_chars=%s", thread_id, len(request), len(answer))


def hybrid_search(asset_id: str, query: str, top_k: int) -> str | None:
    query = clean_text(query)
    try:
        vector = _vectors([query])[0]
    except Exception:
        vector = None
    with connect() as connection:
        if vector is None:
            rows = connection.execute(
                """
                SELECT content FROM asset_chunks
                WHERE asset_id = %s AND tsv @@ plainto_tsquery('simple', %s)
                ORDER BY ts_rank(tsv, plainto_tsquery('simple', %s)) DESC LIMIT %s
                """,
                (asset_id, query, query, top_k),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                WITH keyword AS (
                    SELECT id, content, row_number() OVER (
                        ORDER BY ts_rank(tsv, plainto_tsquery('simple', %s)) DESC
                    ) rank
                    FROM asset_chunks
                    WHERE asset_id = %s AND tsv @@ plainto_tsquery('simple', %s)
                    ORDER BY ts_rank(tsv, plainto_tsquery('simple', %s)) DESC LIMIT %s
                ), semantic AS (
                    SELECT id, content, row_number() OVER (ORDER BY embedding <=> %s::vector) rank
                    FROM asset_chunks WHERE asset_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector LIMIT %s
                ), combined AS (
                    SELECT id, content, rank FROM keyword
                    UNION ALL SELECT id, content, rank FROM semantic
                )
                SELECT content FROM combined GROUP BY id, content
                ORDER BY SUM(1.0 / (60 + rank)) DESC LIMIT %s
                """,
                (query, asset_id, query, query, top_k, str(vector), asset_id, str(vector), top_k, top_k),
            ).fetchall()
    logger.info("Retrieval | database asset=%s mode=%s matches=%s top_k=%s", asset_id, "keyword" if vector is None else "hybrid", len(rows), top_k)
    return "\n\n---\n\n".join(row[0] for row in rows) if rows else None
