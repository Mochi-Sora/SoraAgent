"""On-demand tool discovery.

Only ``find_tools`` (plus ``memory_recall`` when memory is enabled) is visible to
the model at the start of a run.  Every other tool stays registered but hidden
until the model discovers it through ``find_tools``.  The hidden-tool catalog
lives in the ``find_tools`` schema rather than the system prompt, so the prompt
prefix stays small and stable.

Search is deliberately lexical and stdlib-only: token overlap plus
``difflib.SequenceMatcher`` handles exact words and small typos without adding a
dependency or a vector store.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from .tools import Tool, ToolRegistry

FIND_TOOLS_NAME = "find_tools"
MATCH_LIMIT = 5
MIN_SCORE = 2.0
MAX_CATALOG_ENTRIES = 60
MAX_CATALOG_DESCRIPTION = 80
MAX_RESULT_DESCRIPTION = 220

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "from", "get", "give", "help", "how", "i", "in", "is",
    "it", "its", "me", "my", "need", "of", "on", "or", "please", "should", "some",
    "that", "the", "then", "there", "this", "to", "use", "using", "want", "what",
    "when", "where", "which", "with", "would", "you", "your",
})


class ToolGate:
    """What one run may see, and what it has discovered so far."""

    def __init__(
        self,
        registry: ToolRegistry,
        always_visible: Iterable[str] = (),
    ) -> None:
        self.registry = registry
        self.always_visible = frozenset(always_visible)
        self._unlocked: set[str] = set()
        self._catalog: str | None = None

    def visible_names(self) -> list[str]:
        """Registered names sent to the model now, core tools first."""
        tools = self.registry.all()
        always = [
            tool.name
            for tool in tools
            if tool.name in self.always_visible
        ]
        unlocked = [
            tool.name
            for tool in tools
            if tool.name in self._unlocked
            and tool.name not in self.always_visible
        ]
        return always + unlocked

    def visible(self) -> list[Tool]:
        return [self.registry.get(name) for name in self.visible_names()]

    def is_visible(self, name: str) -> bool:
        return name in set(self.visible_names())

    def hidden(self) -> list[Tool]:
        visible = set(self.visible_names())
        return [tool for tool in self.registry.all() if tool.name not in visible]

    def unlock(self, names: Iterable[str]) -> list[str]:
        """Make registered tools visible for the rest of the run."""
        added: list[str] = []
        for name in names:
            if (
                name in self._unlocked
                or name in self.always_visible
                or not self.registry.has(name)
            ):
                continue
            self._unlocked.add(name)
            added.append(name)
        if added:
            self._catalog = None
        return added

    def catalog(self) -> str:
        """Names plus one-line purposes for everything still hidden."""
        if self._catalog is None:
            self._catalog = _render_catalog(self.hidden())
        return self._catalog


class FindToolsTool(Tool):
    """Search the hidden tool library and load the best matches."""

    name = FIND_TOOLS_NAME
    needs_approval = False
    description = (
        "Find and load the tools needed for the current task. The tool list is "
        "partial by design, so call this before concluding that a capability is "
        "unavailable or working around a missing tool. Describe the need in plain "
        "language; matching tools become callable on your next step."
    )

    # This synthetic tool has no callable ``fn`` to pass to ``Tool.__init__``;
    # it overrides ``schema`` and ``call`` directly.
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, gate: ToolGate
    ) -> None:
        self.gate = gate

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
                            "What you want to do, in plain language, e.g. "
                            "'look up a purchase order' or 'expedite an order'."
                        ),
                    }
                },
                "required": ["need"],
            },
        }

    def call(self, need: str = "", **_: Any) -> str:
        query = " ".join(str(need or "").split())
        if not query:
            raise ValueError(
                "find_tools requires a non-empty need, e.g. "
                'find_tools(need="look up a purchase order")'
            )

        matches = self._rank(query)
        if not matches:
            return self._no_match(query)

        loaded = set(self.gate.unlock([candidate.name for _, candidate in matches]))
        return self._render(query, matches, loaded)

    def _rank(self, query: str) -> list[tuple[float, Tool]]:
        scored = [
            (score_need(query, tool.name, tool.description), tool)
            for tool in self.gate.registry.all()
            if tool.name != self.name
        ]
        ranked = [
            (score, tool) for score, tool in scored if score >= MIN_SCORE
        ]
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        return ranked[:MATCH_LIMIT]

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

    def _render(
        self,
        query: str,
        matches: list[tuple[float, Tool]],
        loaded: set[str],
    ) -> str:
        lines = [f"Tools for {query!r}:"]
        for _score, tool in matches:
            if tool.name in loaded:
                state = "now loaded, call it on your next step"
            elif self.gate.is_visible(tool.name):
                state = "already loaded"
            else:
                state = "available"
            description = _clip(
                tool.description, MAX_RESULT_DESCRIPTION
            ).strip()
            lines.append(
                f"- `{tool.name}` ({state}): "
                f"{_signature(tool)} — {description}"
            )
        if loaded:
            lines.append("")
            lines.append(
                "Loaded tools appear in your tool list on the next step; call "
                "them directly."
            )
        return "\n".join(lines)


def score_need(need: str, name: str, description: str) -> float:
    """Rank one registered tool against a plain-language need."""

    need_lower = " ".join(str(need).lower().split())
    name_text = _flatten(name).lower()
    description_lower = " ".join(str(description).lower().split())
    need_tokens = _tokens(need_lower)
    if not need_tokens:
        return 0.0

    name_tokens = _tokens(name_text)
    description_tokens = _tokens(description_lower[:800])
    score = 3.0 * len(need_tokens & name_tokens)
    score += 1.0 * len(need_tokens & (description_tokens - name_tokens))
    score += 4.0 * _ratio(need_lower, name_text)
    score += 1.5 * _ratio(need_lower, description_lower[:300])
    if need_lower in name_text or name_text in need_lower:
        score += 4.0
    return score


def _render_catalog(tools: list[Tool]) -> str:
    if not tools:
        return ""
    shown = tools[:MAX_CATALOG_ENTRIES]
    lines = ["Not loaded yet; call this tool to load what you need.", "", "Tools:"]
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
            type_name = next(
                (item for item in type_name if item != "null"), "any"
            )
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


def _normalize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()
