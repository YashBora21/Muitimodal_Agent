from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import Asset
from .retrieval import chunks, needs_hybrid_retrieval


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
DIRECT_DATABASE_URL = os.getenv("DATABASE_URL_UNPOOLED", DATABASE_URL)
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "database" / "schema.sql"
PAGE_RE = re.compile(r"(?m)^--- Page (\d+) ---$")
_pool = None


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
    with connect() as connection:
        row = connection.execute(
            """
            INSERT INTO users (email, password_hash, display_name)
            VALUES (%s, %s, %s)
            RETURNING id::text, email, COALESCE(display_name, '')
            """,
            (email.lower(), password_hash, display_name),
        ).fetchone()
    return {"id": row[0], "email": row[1], "display_name": row[2]}


def user_by_email(email: str) -> dict[str, str] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id::text, email, password_hash, COALESCE(display_name, '') FROM users WHERE email = %s",
            (email.lower(),),
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
    return assets, [{"role": row[0], "content": row[1]} for row in message_rows]


def _vectors(parts: list[str]) -> list[list[float] | None]:
    try:
        from .retrieval import _embedder

        return _embedder().encode(parts, normalize_embeddings=True).tolist()
    except Exception:
        return [None] * len(parts)


def save_assets(thread_id: str, assets: list[Asset]) -> None:
    with connect() as connection:
        for asset in assets:
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
                    thread_id, asset["id"], asset["kind"], asset["name"],
                    asset.get("content", ""), asset.get("data"),
                    len(asset.get("data", b"")),
                ),
            ).fetchone()
            asset["db_id"] = row[0]
            content = asset.get("content", "")
            if not needs_hybrid_retrieval(asset["kind"], content):
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


def save_messages(thread_id: str, request: str, answer: str) -> None:
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


def hybrid_search(asset_id: str, query: str, top_k: int) -> str | None:
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
    return "\n\n---\n\n".join(row[0] for row in rows) if rows else None



