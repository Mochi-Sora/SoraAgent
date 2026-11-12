"""Web search and page-extraction tools."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Annotated, Any

from ..net import urlopen
from ..tools import Tool, ToolRegistry
from .safety import assert_public_url

_USER_AGENT = "SoraAgent/0.1 (+https://github.com/Mochi-Sora/SoraAgent)"


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> Any:
    body = None
    request_headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method="POST" if body else "GET"
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_000)
    return json.loads(raw.decode("utf-8", errors="replace"))


def _search_results(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("results")
    if not isinstance(candidates, list):
        web = payload.get("web")
        candidates = web.get("results") if isinstance(web, dict) else []
    out: list[dict[str, str]] = []
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "title": str(item.get("title") or item.get("name") or "").strip(),
                "url": str(item.get("url") or item.get("link") or "").strip(),
                "snippet": str(
                    item.get("content")
                    or item.get("description")
                    or item.get("snippet")
                    or ""
                ).strip(),
            }
        )
    return [item for item in out if item["url"]]


def _render_results(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web results for {query!r}."
    lines = [f"Web results for {query!r}:"]
    for index, item in enumerate(results, 1):
        lines.append("")
        lines.append("{}. {}".format(index, item["title"] or "(untitled)"))
        lines.append("   {}".format(item["url"]))
        if item["snippet"]:
            lines.append("   {}".format(item["snippet"][:500]))
    return "\n".join(lines)


class _HTMLText(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "template", "svg"})
    _BLOCK = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "li",
            "tr",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "pre",
            "blockquote",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in self._SKIP:
            self._skip_depth += 1
        elif lower == "title":
            self._in_title = True
        elif lower in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif lower == "title":
            self._in_title = False
        elif lower in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip_depth:
            return
        text = " ".join(str(data).split())
        if text:
            self.parts.append(text + " ")

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()


def register_web_tools(registry: ToolRegistry, config: Any) -> list[Tool]:
    added: list[Tool] = []

    def web_search(
        query: Annotated[str, "Search query"],
        limit: Annotated[int, "Maximum results (1-10)"] = 5,
    ) -> str:
        count = max(1, min(int(limit), 10, config.search_results))
        timeout = config.web_timeout
        try:
            tavily = os.getenv("TAVILY_API_KEY", "").strip()
            brave = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
            custom = os.getenv("SORA_WEB_SEARCH_URL", "").strip()
            if tavily:
                payload = _http_json(
                    "https://api.tavily.com/search",
                    payload={
                        "api_key": tavily,
                        "query": query,
                        "max_results": count,
                        "include_answer": False,
                    },
                    timeout=timeout,
                )
            elif brave:
                url = (
                    "https://api.search.brave.com/res/v1/web/search?"
                    + urllib.parse.urlencode({"q": query, "count": count})
                )
                payload = _http_json(
                    url,
                    headers={
                        "X-Subscription-Token": brave,
                        "Accept": "application/json",
                    },
                    timeout=timeout,
                )
            elif custom:
                headers = {}
                key = os.getenv("SORA_WEB_SEARCH_API_KEY", "").strip()
                if key:
                    headers["Authorization"] = "Bearer " + key
                payload = _http_json(
                    custom,
                    payload={"query": query, "limit": count},
                    headers=headers,
                    timeout=timeout,
                )
            else:
                return (
                    "error: web_search is not configured. Set TAVILY_API_KEY, "
                    "BRAVE_SEARCH_API_KEY, or SORA_WEB_SEARCH_URL."
                )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return f"error: web search failed: {exc}"
        results = _search_results(payload)
        return _render_results(query, results[:count])

    def web_extract(
        url: Annotated[str, "HTTP(S) page URL"],
        max_chars: Annotated[int, "Maximum characters to return"] = 15000,
    ) -> str:
        try:
            target = assert_public_url(url, allow_private=config.allow_private_network)
        except ValueError as exc:
            return f"error: {exc}"
        limit = max(500, min(int(max_chars), config.web_max_chars))
        request = urllib.request.Request(
            target,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            },
        )
        try:
            with urlopen(request, timeout=config.web_timeout) as response:
                final_url = response.geturl()
                assert_public_url(final_url, allow_private=config.allow_private_network)
                raw = response.read(max(4_000, limit * 4))
                charset = response.headers.get_content_charset() or "utf-8"
                content_type = response.headers.get("Content-Type", "")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return f"error: could not fetch {target}: {exc}"
        text = raw.decode(charset, errors="replace")
        if "html" in content_type.lower() or re.search(
            r"<html|<body|<p[\s>]", text[:2000], re.IGNORECASE
        ):
            parser = _HTMLText()
            parser.feed(text)
            title = parser.title.strip()
            body = parser.text()
        else:
            title = ""
            body = text
        body = body[: max(0, limit - 80)].rstrip()
        if len(body) >= limit - 80:
            body += "\n...[truncated]"
        lines = [f"URL: {final_url}"]
        if title:
            lines.append(f"Title: {title}")
        lines.append("")
        lines.append(body or "(no readable text)")
        return "\n".join(lines)

    definitions = [
        Tool(
            web_search,
            "web_search",
            "Search the web for current information. Requires a configured search backend.",
            aliases=("search web", "internet search", "google", "tavily", "brave"),
        ),
        Tool(
            web_extract,
            "web_extract",
            "Fetch a public HTTP(S) page and return readable text. Use instead of raw curl.",
            aliases=("extract url", "read webpage", "fetch url", "browse page"),
        ),
    ]
    for definition in definitions:
        if not registry.has(definition.name):
            registry.add(definition)
            added.append(definition)
    return added
