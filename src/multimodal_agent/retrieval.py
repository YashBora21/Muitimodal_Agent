from __future__ import annotations

from functools import cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RAG_MIN_CHARS,
    WHOLE_DOCUMENT_TERMS,
)


def needs_hybrid_retrieval(content: str) -> bool:
    return len(content) > RAG_MIN_CHARS


def retrieval_scope(query: str, requested_scope: str = "auto") -> str:
    if requested_scope in {"focused", "whole"}:
        return requested_scope
    normalized = " ".join(query.lower().split())
    return (
        "whole"
        if any(term in normalized for term in WHOLE_DOCUMENT_TERMS)
        else "focused"
    )


def chunks(content: str) -> list[str]:
    if not content.strip():
        return []
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_text(content)


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)
