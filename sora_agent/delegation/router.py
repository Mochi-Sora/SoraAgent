"""Dependency-free lexical routing for specialists."""

from __future__ import annotations

import math
import re
from collections import Counter

from .model import DelegationAgent

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def rank(
    agents: list[DelegationAgent], need: str, limit: int = 3
) -> list[tuple[DelegationAgent, float]]:
    """Rank active specialists against a plain-language need.

    The corpus is intentionally tiny, so a BM25-style score with a small
    substring boost is more predictable than an embedding call.
    """

    query = tokens(need)
    if not query or not agents:
        return []
    docs = [tokens(agent.routing_text()) for agent in agents]
    document_frequency = Counter()
    for doc in docs:
        for term in set(doc):
            document_frequency[term] += 1
    total = len(agents)
    average_length = sum(len(doc) for doc in docs) / max(1, total)
    scored: list[tuple[DelegationAgent, float]] = []
    lowered = str(need or "").lower()
    for agent, doc in zip(agents, docs):
        if not doc:
            continue
        counts = Counter(doc)
        score = 0.0
        for term in set(query):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(doc) / max(1.0, average_length)
            )
            score += idf * (frequency * 2.2) / denominator
        if agent.name.lower() in lowered:
            score += 2.0
        if agent.when_to_use and agent.when_to_use.lower() in lowered:
            score += 1.5
        if score > 0:
            scored.append((agent, score))
    scored.sort(key=lambda item: (-item[1], item[0].id))
    return scored[: max(1, min(int(limit), 10))]


def best(agents: list[DelegationAgent], need: str) -> DelegationAgent | None:
    matches = rank(agents, need, limit=1)
    return matches[0][0] if matches else None
