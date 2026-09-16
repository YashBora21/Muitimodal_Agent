from __future__ import annotations

import math
import re
from collections import Counter
from functools import cache

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RAG_MIN_CHARS,
    RAG_MIN_PDF_PAGES,
    RETRIEVAL_TOP_K,
)


TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
PDF_PAGE_RE = re.compile(r"(?m)^--- Page \d+ ---$")


def needs_hybrid_retrieval(kind: str, content: str) -> bool:
    if len(content) <= RAG_MIN_CHARS:
        return False
    if kind == "pdf":
        return len(PDF_PAGE_RE.findall(content)) > RAG_MIN_PDF_PAGES
    return kind in {"audio", "youtube"}


def chunks(content: str) -> list[str]:
    if len(content) <= CHUNK_SIZE:
        return [content] if content.strip() else []

    result = []
    start = 0
    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        if end < len(content):
            boundary = content.rfind("\n", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        text = content[start:end].strip()
        if text:
            result.append(text)
        if end >= len(content):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return result


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _keyword_ranks(parts: list[str], query: str) -> list[int]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    documents = [_tokens(part) for part in parts]
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    average_length = sum(map(len, documents)) / max(len(documents), 1)
    scores = []
    for index, document in enumerate(documents):
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * len(document) / max(average_length, 1)
            )
            score += inverse_frequency * frequency * 2.5 / denominator
        if score:
            scores.append((score, index))
    return [index for _, index in sorted(scores, reverse=True)]


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _semantic_ranks(parts: list[str], query: str) -> list[int]:
    try:
        vectors = _embedder().encode(
            [query, *parts],
            normalize_embeddings=True,
        )
    except Exception:
        return []
    scores = vectors[1:] @ vectors[0]
    return sorted(range(len(parts)), key=lambda index: scores[index], reverse=True)


def hybrid_search(content: str, query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
    parts = chunks(content)
    if not parts:
        return "(empty)"
    if len(parts) == 1:
        return parts[0]

    rankings = (_keyword_ranks(parts, query), _semantic_ranks(parts, query))
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking[:top_k], 1):
            scores[index] = scores.get(index, 0.0) + 1 / (60 + rank)

    if not scores:
        return "No content matched that query."
    selected = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return "\n\n---\n\n".join(parts[index] for index in selected)
