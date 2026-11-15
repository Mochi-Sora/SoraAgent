"""Web search and page-extraction tools."""

from __future__ import annotations

import json
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


_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_DDG_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def _decode_ddg_href(href: str) -> str:
    value = str(href or "").strip()
    if not value:
        return ""
    if value.startswith("//"):
        value = "https:" + value
    parsed = urllib.parse.urlsplit(value)
    if parsed.path.startswith("/l/") and (
        not parsed.netloc or parsed.netloc.lower().endswith("duckduckgo.com")
    ):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return target
    return value


class _DuckDuckGoParser(HTMLParser):
    """Extract ``result__a`` links and ``result__snippet`` text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._active: dict[str, str] | None = None
        self._mode: str | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            url = _decode_ddg_href(values.get("href") or "")
            if url and "duckduckgo.com/y.js" not in url:
                self._active = {"title": "", "url": url, "snippet": ""}
                self.results.append(self._active)
                self._mode = "title"
                self._depth = 1
            else:
                self._active = None
                self._mode = None
            return
        if (
            self._active is not None
            and self._mode is None
            and "result__snippet" in classes
        ):
            self._mode = "snippet"
            self._depth = 1
            return
        if self._mode is not None and tag not in _DDG_VOID:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._mode is None or tag in _DDG_VOID:
            return
        self._depth -= 1
        if self._depth <= 0:
            self._mode = None
            self._depth = 0

    def handle_data(self, data: str) -> None:
        if self._active is None or self._mode is None:
            return
        if self._mode == "title":
            self._active["title"] += data
        else:
            self._active["snippet"] += data


def _parse_ddg_results(html: str, limit: int) -> list[dict[str, str]]:
    parser = _DuckDuckGoParser()
    parser.feed(html)
    results: list[dict[str, str]] = []
    for item in parser.results:
        title = " ".join(item["title"].split())
        snippet = " ".join(item["snippet"].split())
        if title and item["url"]:
            results.append(
                {"title": title, "url": item["url"], "snippet": snippet}
            )
    return results[:limit]


def _duckduckgo_search(query: str, limit: int, timeout: float) -> list[dict[str, str]]:
    url = _DDG_HTML_URL + "?" + urllib.parse.urlencode(
        {"q": query, "kl": "wt-wt"}
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _DDG_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        raw = response.read(2_000_000)
        charset = response.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, errors="replace")
    if status != 200:
        raise ValueError(
            f"DuckDuckGo returned HTTP {status} (challenge or rate limit)"
        )
    results = _parse_ddg_results(text, limit)
    if not results and ("anomaly" in text.lower() or "challenge" in text.lower()):
        raise ValueError(
            "DuckDuckGo returned a bot challenge; use Tavily, Brave, or a "
            "custom endpoint for this server"
        )
    return results


def register_web_tools(registry: ToolRegistry, config: Any) -> list[Tool]:
    added: list[Tool] = []

    def web_search(
        query: Annotated[str, "Search query"],
        limit: Annotated[int, "Maximum results (1-10)"] = 5,
    ) -> str:
        count = max(1, min(int(limit), 10, config.search_results))
        timeout = config.web_timeout
        search = config.web_search
        provider = search.resolved_provider()
        try:
            if provider == "off":
                return (
                    "error: web_search is disabled by configuration "
                    "(features.web_search.provider = 'off')."
                )
            if provider == "duckduckgo":
                results = _duckduckgo_search(query, count, timeout)
                return _render_results(query, results[:count])
            if provider == "tavily":
                if not search.tavily_api_key:
                    return (
                        "error: web_search provider 'tavily' is selected but "
                        "TAVILY_API_KEY is not set (environment or secrets.env)."
                    )
                payload = _http_json(
                    "https://api.tavily.com/search",
                    payload={
                        "api_key": search.tavily_api_key,
                        "query": query,
                        "max_results": count,
                        "include_answer": False,
                    },
                    timeout=timeout,
                )
            elif provider == "brave":
                if not search.brave_api_key:
                    return (
                        "error: web_search provider 'brave' is selected but "
                        "BRAVE_SEARCH_API_KEY is not set (environment or secrets.env)."
                    )
                url = (
                    "https://api.search.brave.com/res/v1/web/search?"
                    + urllib.parse.urlencode({"q": query, "count": count})
                )
                payload = _http_json(
                    url,
                    headers={
                        "X-Subscription-Token": search.brave_api_key,
                        "Accept": "application/json",
                    },
                    timeout=timeout,
                )
            elif provider == "custom":
                if not search.url:
                    return (
                        "error: web_search provider 'custom' is selected but no "
                        "SORA_WEB_SEARCH_URL / features.web_search.url is set."
                    )
                headers = {}
                if search.api_key:
                    headers["Authorization"] = "Bearer " + search.api_key
                payload = _http_json(
                    search.url,
                    payload={"query": query, "limit": count},
                    headers=headers,
                    timeout=timeout,
                )
            else:
                return f"error: unknown web_search provider: {provider!r}"
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            if provider == "duckduckgo":
                return (
                    "error: DuckDuckGo search failed: "
                    f"{exc}. If this server is blocked, set "
                    "features.web_search.provider to tavily, brave, or custom."
                )
            return f"error: web search ({provider}) failed: {exc}"
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
