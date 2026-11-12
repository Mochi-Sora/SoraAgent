"""On-demand tool discovery.

Only ``find_tools`` (plus ``memory_recall`` when memory is enabled) is visible to
the model at the start of a run.  Every other tool stays registered but hidden
until the model discovers it through ``find_tools``.  The hidden-tool catalog
lives in the ``find_tools`` schema rather than the system prompt, so the prompt
prefix stays small and stable.

Discovery has two tiers::

* ``find_tools(need="...")`` previews the closest matches without loading them.
  A single clearly dominant match loads automatically so the common case stays
  one step; ambiguous searches return a shortlist.
* ``find_tools(load=["name", ...])`` loads exact names, at most ``LOAD_LIMIT``
  per call.  ``ToolGate`` also caps how many hidden tools one run may expose.

Search is deliberately lexical and stdlib-only: token overlap with IDF weights,
name boosts, optional tool aliases, and ``difflib.SequenceMatcher`` handle exact
words and small typos without adding a dependency or a vector store.  The
``ToolRanker`` protocol keeps the ranking strategy swappable for a future
embedding backend.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Protocol

from .tools import Tool, ToolRegistry

FIND_TOOLS_NAME = "find_tools"
PREVIEW_LIMIT = 8
LOAD_LIMIT = 3
MIN_SCORE = 2.0
AUTO_LOAD_SCORE = 8.0
AUTO_LOAD_MARGIN = 2.5
MAX_VISIBLE_TOOLS = 12
MAX_CATALOG_ENTRIES = 40
MAX_CATALOG_DESCRIPTION = 50
MAX_RESULT_DESCRIPTION = 160

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "get",
        "give",
        "help",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "need",
        "of",
        "on",
        "or",
        "please",
        "should",
        "some",
        "that",
        "the",
        "then",
        "there",
        "this",
        "to",
        "use",
        "using",
        "want",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class Candidate:
    """One ranked tool and its lexical score."""

    tool: Tool
    score: float


class ToolRanker(Protocol):
    """Rank registered tools against a plain-language need."""

    def rank(self, need: str, tools: Sequence[Tool], limit: int) -> list[Candidate]: ...


class ToolGate:
    """What one run may see, and what it has discovered so far."""

    def __init__(
        self,
        registry: ToolRegistry,
        always_visible: Iterable[str] = (),
        max_visible: int | None = MAX_VISIBLE_TOOLS,
    ) -> None:
        self.registry = registry
        self.always_visible = frozenset(always_visible)
        self.max_visible = None if max_visible is None else max(0, int(max_visible))
        self._unlocked: set[str] = set()
        self._catalog: str | None = None

    def visible_names(self) -> list[str]:
        """Registered names sent to the model now, core tools first."""
        tools = self.registry.all()
        always = [tool.name for tool in tools if tool.name in self.always_visible]
        unlocked = [
            tool.name
            for tool in tools
            if tool.name in self._unlocked and tool.name not in self.always_visible
        ]
        return always + unlocked

    def visible(self) -> list[Tool]:
        return [self.registry.get(name) for name in self.visible_names()]

    def is_visible(self, name: str) -> bool:
        return name in set(self.visible_names())

    def hidden(self) -> list[Tool]:
        visible = set(self.visible_names())
        return [tool for tool in self.registry.all() if tool.name not in visible]

    def remaining_slots(self) -> int | None:
        """How many more hidden tools this run may expose, if capped."""
        if self.max_visible is None:
            return None
        return max(0, self.max_visible - len(self.visible_names()))

    def unlock(self, names: Iterable[str]) -> list[str]:
        """Make registered tools visible for the rest of the run."""
        added: list[str] = []
        remaining = self.remaining_slots()
        for name in names:
            if remaining is not None and remaining <= 0:
                break
            if (
                name in self._unlocked
                or name in self.always_visible
                or not self.registry.has(name)
            ):
                continue
            self._unlocked.add(name)
            added.append(name)
            if remaining is not None:
                remaining -= 1
        return added

    def catalog(self) -> str:
        """Stable names-plus-purpose catalog for the whole run.

        The catalog deliberately includes already-loaded tools.  Removing them
        on unlock would rewrite the ``find_tools`` schema after every load and
        invalidate the provider's tool-definition prefix.
        """
        if self._catalog is None:
            tools = [
                tool
                for tool in self.registry.all()
                if tool.name != "find_tools" and tool.name not in self.always_visible
            ]
            self._catalog = _render_catalog(tools)
        return self._catalog


class LexicalRanker:
    """Deterministic lexical ranking with IDF weights and tool aliases."""

    def rank(self, need: str, tools: Sequence[Tool], limit: int) -> list[Candidate]:
        query = _normalize(need)
        tokens = _tokens(query)
        if not tokens or limit <= 0:
            return []

        documents = [_document(tool) for tool in tools]
        if not documents:
            return []

        counts: dict[str, int] = {}
        for document in documents:
            for token in document.all_tokens & tokens:
                counts[token] = counts.get(token, 0) + 1
        total = len(documents)

        def idf(token: str) -> float:
            return math.log((total + 1) / (counts.get(token, 0) + 1)) + 1.0

        scored = [
            (_score(query, tokens, document, idf), document.tool)
            for document in documents
        ]
        ranked = sorted(
            ((score, tool) for score, tool in scored if score >= MIN_SCORE),
            key=lambda item: (-item[0], item[1].name),
        )
        return [Candidate(tool=tool, score=score) for score, tool in ranked[:limit]]


class FindToolsTool(Tool):
    """Preview the hidden tool library and load the tools a task needs."""

    name = FIND_TOOLS_NAME
    needs_approval = False
    description = (
        "Discover tools. Call with `need` in plain language to preview the "
        "closest matches; the preview does not load them. Then call with "
        "`load=[name, ...]` to make up to "
        f"{LOAD_LIMIT} tools callable, or use a precise need when one match is "
        "clearly dominant and it will load automatically. The tool list is "
        "partial; call this before assuming a capability is missing."
    )

    # This synthetic tool has no callable ``fn`` to pass to ``Tool.__init__``;
    # it overrides ``schema`` and ``call`` directly.
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, gate: ToolGate, ranker: ToolRanker | None = None
    ) -> None:
        self.gate = gate
        self.ranker: ToolRanker = ranker or LexicalRanker()

    @property
    def schema(self) -> dict[str, Any]:
        description = self.description
        catalog = self.gate.catalog()
        if catalog:
            description = f"{description}\n\n{catalog}"
        return {
            "name": self.name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "need": {
                        "type": "string",
                        "description": (
                            "What you need, in plain language. A search only "
                            "previews matches."
                        ),
                    },
                    "load": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"Exact tool names to load now (max {LOAD_LIMIT} per call)."
                        ),
                    },
                },
                "required": [],
            },
        }

    def call(self, need: str = "", load: Any = None, **_: Any) -> str:
        names = _parse_load(load)
        query = _normalize(need)
        if not names and not query:
            raise ValueError(
                "find_tools requires `need` or `load`, e.g. "
                'find_tools(need="look up a purchase order") or '
                'find_tools(load=["lookup_order"])'
            )
        if names:
            return self._load(names)
        return self._search(query)

    # ------------------------------------------------------------------
    # Tier 1: preview
    # ------------------------------------------------------------------

    def _search(self, query: str) -> str:
        candidates = self._rank(query)
        if not candidates:
            return self._no_match(query)

        top = candidates[0]
        if not self.gate.is_visible(top.tool.name) and _dominant(candidates):
            loaded = self.gate.unlock([top.tool.name])
            if loaded:
                return self._render_loaded(
                    query=query,
                    loaded=loaded,
                    already=[],
                    unknown=[],
                    overflow=[],
                    budget=[],
                )
        return self._render_preview(query, candidates)

    def _rank(self, query: str) -> list[Candidate]:
        tools = [tool for tool in self.gate.registry.all() if tool.name != self.name]
        return self.ranker.rank(query, tools, PREVIEW_LIMIT)

    # ------------------------------------------------------------------
    # Tier 2: explicit load
    # ------------------------------------------------------------------

    def _load(self, names: list[str]) -> str:
        resolved, unknown = self._resolve(names)
        already = [name for name in resolved if self.gate.is_visible(name)]
        pending = [name for name in resolved if not self.gate.is_visible(name)]
        requested = pending[:LOAD_LIMIT]
        overflow = pending[LOAD_LIMIT:]
        loaded = self.gate.unlock(requested)
        budget = [name for name in requested if name not in loaded]
        return self._render_loaded(
            query="",
            loaded=loaded,
            already=already,
            unknown=unknown,
            overflow=overflow,
            budget=budget,
        )

    def _resolve(self, names: list[str]) -> tuple[list[str], list[str]]:
        """Resolve exact, case-insensitive, and flattened tool names."""
        lookup: dict[str, str] = {}
        for tool in self.gate.registry.all():
            if tool.name == self.name:
                continue
            lookup[tool.name.lower()] = tool.name
            lookup.setdefault(_normalize(_flatten(tool.name)), tool.name)

        resolved: list[str] = []
        unknown: list[str] = []
        for raw in names:
            key = raw.strip()
            if not key:
                continue
            match = lookup.get(key.lower()) or lookup.get(_normalize(_flatten(key)))
            if match is None:
                unknown.append(key)
                continue
            if match not in resolved:
                resolved.append(match)
        return resolved, unknown

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_preview(self, query: str, candidates: list[Candidate]) -> str:
        lines = [f"Tools for {query!r} (preview only):"]
        for candidate in candidates:
            tool = candidate.tool
            state = "already loaded" if self.gate.is_visible(tool.name) else "available"
            description = _clip(tool.description, MAX_RESULT_DESCRIPTION).strip()
            lines.append(
                f"- `{tool.name}` ({state}): {_signature(tool)} — {description}"
            )
        names = [
            candidate.tool.name
            for candidate in candidates
            if not self.gate.is_visible(candidate.tool.name)
        ][:LOAD_LIMIT]
        remaining = self.gate.remaining_slots()
        if names and remaining == 0:
            lines.append("")
            lines.append(
                "Visible-tool budget reached "
                f"({self.gate.max_visible} per run); use the tools already "
                "loaded."
            )
        elif names:
            quoted = ", ".join(f'"{name}"' for name in names)
            lines.append("")
            lines.append(
                f"Nothing was loaded. To load up to {LOAD_LIMIT}, call "
                f"find_tools(load=[{quoted}])."
            )
        elif candidates:
            lines.append("")
            lines.append("Every match is already loaded; call it directly.")
        return "\n".join(lines)

    def _render_loaded(
        self,
        *,
        query: str,
        loaded: list[str],
        already: list[str],
        unknown: list[str],
        overflow: list[str],
        budget: list[str],
    ) -> str:
        heading = f"Loaded tools for {query!r}:" if query else "Loaded tools:"
        lines = [heading]
        if loaded:
            lines.append(
                ", ".join(f"`{name}`" for name in loaded) + " — callable next step."
            )
        else:
            lines.append("No new tools were loaded.")
        if already:
            lines.append(
                "Already loaded: " + ", ".join(f"`{name}`" for name in already) + "."
            )
        if unknown:
            suggestions = [_suggest(name, self.gate.registry.all()) for name in unknown]
            detail = ", ".join(
                f"`{name}`" + (f" (did you mean {hint}?)" if hint else "")
                for name, hint in zip(unknown, suggestions)
            )
            lines.append(f"Unknown: {detail}.")
        if overflow:
            lines.append(
                f"Not loaded (max {LOAD_LIMIT} per call): "
                + ", ".join(f"`{name}`" for name in overflow)
                + ". Call find_tools again for the rest."
            )
        if budget:
            lines.append(
                "Visible-tool budget reached "
                f"({self.gate.max_visible} per run): "
                + ", ".join(f"`{name}`" for name in budget)
                + ". Finish this turn with the tools already loaded."
            )
        return "\n".join(lines)

    def _no_match(self, query: str) -> str:
        names = [tool.name for tool in self.gate.hidden()]
        lines = [f"No tool matched {query!r}."]
        if names:
            listed = ", ".join(f"`{name}`" for name in names[:40])
            lines.append(f"Available tools: {listed}.")
            if len(names) > 40:
                lines.append(f"({len(names) - 40} more not listed.)")
            lines.append(
                "Try different words, or continue with the tools already in "
                "your tool list."
            )
        else:
            lines.append("All tools are already loaded.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Document:
    tool: Tool
    name_text: str
    description_text: str
    alias_texts: tuple[str, ...]
    name_tokens: frozenset[str]
    description_tokens: frozenset[str]
    alias_tokens: frozenset[str]

    @property
    def all_tokens(self) -> frozenset[str]:
        return self.name_tokens | self.description_tokens | self.alias_tokens


def _document(tool: Tool) -> _Document:
    aliases = tuple(str(item) for item in getattr(tool, "aliases", ()))
    name_text = _flatten(tool.name).lower()
    description_text = _normalize(tool.description)
    alias_texts = tuple(_normalize(alias) for alias in aliases)
    return _Document(
        tool=tool,
        name_text=name_text,
        description_text=description_text,
        alias_texts=alias_texts,
        name_tokens=frozenset(_tokens(name_text)),
        description_tokens=frozenset(_tokens(description_text[:800])),
        alias_tokens=frozenset(_tokens(" ".join(alias_texts))),
    )


def _score(
    query: str,
    tokens: frozenset[str],
    document: _Document,
    idf: Callable[[str], float],
) -> float:
    name_hits = tokens & document.name_tokens
    description_hits = (tokens & document.description_tokens) - document.name_tokens
    alias_hits = (tokens & document.alias_tokens) - document.name_tokens

    score = 3.0 * sum(idf(token) for token in name_hits)
    score += 1.0 * sum(idf(token) for token in description_hits)
    score += 2.0 * sum(idf(token) for token in alias_hits)
    score += 4.0 * _ratio(query, document.name_text)
    score += 1.5 * _ratio(query, document.description_text[:300])
    if len(query) >= 4 and (query in document.name_text or document.name_text in query):
        score += 4.0
    name_key = query.replace(" ", "_")
    if name_key == document.tool.name.lower() or query in document.alias_texts:
        score += 6.0
    return score


def score_need(
    need: str, name: str, description: str, aliases: Iterable[str] = ()
) -> float:
    """Score one tool against a need, without corpus-level IDF weights."""

    query = _normalize(need)
    tokens = _tokens(query)
    if not tokens:
        return 0.0
    alias_texts = tuple(_normalize(alias) for alias in aliases)
    document = _Document(
        tool=Tool(lambda: None, name, description),
        name_text=_flatten(name).lower(),
        description_text=_normalize(description),
        alias_texts=alias_texts,
        name_tokens=frozenset(_tokens(_flatten(name).lower())),
        description_tokens=frozenset(_tokens(_normalize(description)[:800])),
        alias_tokens=frozenset(_tokens(" ".join(alias_texts))),
    )
    return _score(query, tokens, document, lambda _token: 1.0)


def _dominant(candidates: list[Candidate]) -> bool:
    """Whether the top match is strong and clearly ahead of the runner-up."""
    top = candidates[0]
    if top.score < AUTO_LOAD_SCORE:
        return False
    if len(candidates) == 1:
        return True
    return top.score - candidates[1].score >= AUTO_LOAD_MARGIN


def _parse_load(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = raw.replace(",", " ").split()
    elif isinstance(raw, (list, tuple, set)):
        items = [str(item) for item in raw]
    else:
        raise TypeError("find_tools `load` must be a list of tool names")
    names: list[str] = []
    for item in items:
        name = " ".join(str(item).split())
        if name and name not in names:
            names.append(name)
    return names


def _suggest(name: str, tools: Sequence[Tool]) -> str:
    matches = get_close_matches(
        name.lower(),
        [tool.name.lower() for tool in tools if tool.name != FIND_TOOLS_NAME],
        n=1,
        cutoff=0.6,
    )
    return f"`{matches[0]}`" if matches else ""


def _render_catalog(tools: list[Tool]) -> str:
    if not tools:
        return ""
    shown = tools[:MAX_CATALOG_ENTRIES]
    lines = ["Tool catalog; use find_tools to load what you need:"]
    lines.extend(
        f"- `{tool.name}`: {_clip(tool.description, MAX_CATALOG_DESCRIPTION)}"
        for tool in shown
    )
    omitted = len(tools) - len(shown)
    if omitted > 0:
        lines.append("")
        lines.append(f"({omitted} more not listed; search with a specific need.)")
    return "\n".join(lines)


def _signature(tool: Tool) -> str:
    """Compact ``name(required: type, optional?: type)`` call shape."""

    try:
        schema = tool.schema
    except Exception:  # noqa: BLE001 - a broken schema should not break search
        return f"{tool.name}()"
    parameters = schema.get("parameters") if isinstance(schema, dict) else None
    if not isinstance(parameters, dict):
        return f"{tool.name}()"
    properties = parameters.get("properties")
    if not isinstance(properties, dict) or not properties:
        return f"{tool.name}()"
    required = parameters.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    parts: list[str] = []
    for parameter, raw in properties.items():
        fragment = raw if isinstance(raw, dict) else {}
        type_name = fragment.get("type") or "any"
        if isinstance(type_name, list):
            type_name = next((item for item in type_name if item != "null"), "any")
        marker = "" if parameter in required_names else "?"
        parts.append(f"{parameter}{marker}: {type_name}")
    return f"{tool.name}({', '.join(parts)})"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3].rstrip() + "..."


def _flatten(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _normalize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        {
            _normalize_token(token)
            for token in _TOKEN_RE.findall(text.lower())
            if len(token) > 1 and token not in _STOPWORDS
        }
    )


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()
