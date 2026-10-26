"""A minimal RAG interface with an in-memory stub.

Swap `MemoryRetriever` for a real vector DB (e.g. Azure AI Search, Qdrant, pgvector) and keep
the same `retrieve()` contract. Everything else in the harness is RAG-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        ...


class MemoryRetriever:
    def __init__(self, documents: Iterable[str]) -> None:
        self.docs = list(documents)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        # Tiny keyword-match fallback so it runs offline; use embeddings in production.
        q = query.lower()
        scored = []
        for d in self.docs:
            hits = sum(1 for tok in q.split() if tok and tok in d.lower())
            scored.append((hits, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]


def build_prompt(query: str, retrieved: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(retrieved))
    return (
        "Use ONLY the context below to answer. If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
