"""Entity and relation extraction for CortexDB's native memory graph.

CortexDB can build graph nodes for memories when ``save_memory`` receives
``entities`` (and optionally ``relations``).  It also exposes a deterministic
``extract_conversation`` tool.  We prefer that tool during dream transfer and
fall back to a small local extractor when a backend does not implement tools
(for example the in-memory test backend).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[-_][A-Z0-9]+)+\b")
_PROPER_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,2}\b"
)
_REDACTION_MARKERS = ("[profile]", "[profile details withheld]")

_STOPWORDS = frozenset(
    (
        "a", "an", "the", "i", "we", "you", "he", "she", "it", "they",
        "this", "that", "these", "those", "please", "thanks", "thank",
        "hi", "hello", "yes", "no", "but", "and", "or", "if", "when",
        "where", "what", "who", "why", "how", "can", "could", "would",
        "should", "will", "may", "might", "must", "do", "does", "did",
        "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "not", "now", "then", "there", "here", "also",
        "just", "only", "very", "more", "most", "some", "any", "all",
        "each", "every", "other", "another", "such", "own", "same", "so",
        "than", "too", "as", "at", "by", "for", "from", "in", "into",
        "of", "on", "onto", "out", "over", "per", "to", "up", "via",
        "with", "without", "about", "after", "before", "during", "under",
        "again", "further", "once", "check", "look", "find", "show",
        "give", "tell", "use", "make", "get", "set", "run", "send",
        "read", "write", "update", "create", "delete", "remove", "open",
        "close", "start", "stop", "deploy", "expedite", "cancel", "status",
        "details", "order", "purchase", "release", "date", "owner",
    )
)


def extract_entities(text: str, limit: int = 12) -> list[str]:
    """Extract identifier-like and proper-noun entities from ``text``.

    This is deliberately small and deterministic.  It is the fallback for
    backends that cannot run CortexDB's ``extract_conversation`` tool.
    """

    value = str(text or "")
    if not value.strip():
        return []
    identifiers = _IDENTIFIER_RE.findall(value)
    identifier_keys = {item.casefold() for item in identifiers}
    seen = {item.casefold() for item in identifiers}
    out = list(identifiers)
    for match in _PROPER_RE.findall(value):
        words = match.strip().split()
        while words and words[0].casefold() in _STOPWORDS:
            words.pop(0)
        if not words:
            continue
        candidate = " ".join(words)
        key = candidate.casefold()
        if key in seen or key in _STOPWORDS:
            continue
        if any(
            key == item or key + "-" in item or key + "_" in item
            for item in identifier_keys
        ):
            continue
        if any(marker in candidate for marker in _REDACTION_MARKERS):
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max(1, int(limit)):
            break
    return out[: max(1, int(limit))]


def graph_hints(
    backend: Any, text: str, limit: int = 12
) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(entities, relations)`` for a memory or a recall query.

    The backend's own extraction is preferred.  Any failure falls back to the
    local extractor and never breaks dream or recall.
    """

    call_tool = getattr(backend, "call_tool", None)
    if callable(call_tool):
        try:
            response = call_tool(
                "extract_conversation",
                {"text": str(text or ""), "persist": False, "max_entities": limit},
            )
        except Exception:  # noqa: BLE001 - graph extraction is optional
            response = None
        if isinstance(response, dict):
            names = _clean_names(response.get("entities"), limit)
            relations = _clean_relations(response.get("relations"))
            if names:
                return names, relations
    return extract_entities(text, limit), []


def entity_payloads(names: Iterable[str]) -> list[dict[str, str]]:
    return [{"name": name, "type": "entity"} for name in _clean_names(names, 100)]


def relation_payloads(relations: Iterable[Any]) -> list[dict[str, str]]:
    return _clean_relations(relations)


def _clean_names(values: Any, limit: int) -> list[str]:
    if isinstance(values, (str, bytes)):
        return []
    try:
        items = list(values)
    except TypeError:
        return []
    out: list[str] = []
    seen = set()
    for raw in items:
        name = " ".join(str(raw or "").split())
        key = name.casefold()
        if not name or len(name) < 2 or key in seen:
            continue
        if key in {"profile", "profile details withheld"}:
            continue
        if "[" in name or "]" in name:
            continue
        if any(marker in name for marker in _REDACTION_MARKERS):
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _clean_relations(values: Any) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)):
        return []
    try:
        items = list(values)
    except TypeError:
        return []
    out: list[dict[str, str]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        left = " ".join(str(item.get("from") or "").split())
        right = " ".join(str(item.get("to") or "").split())
        kind = " ".join(str(item.get("type") or "co_occurs").split())
        if not left or not right or left.casefold() == right.casefold():
            continue
        if any(marker in left or marker in right for marker in _REDACTION_MARKERS):
            continue
        key = (left.casefold(), right.casefold(), kind.casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append({"from": left, "to": right, "type": kind or "co_occurs"})
    return out
