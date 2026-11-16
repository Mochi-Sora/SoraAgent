"""Plain text, CSV/TSV, and HTML adapters."""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from typing import ClassVar

from ..model import Document
from ..sniff import (
    LANGUAGE_BY_SUFFIX,
    TEXT_EXTENSIONS,
    decode_text,
    html_like,
    looks_like_text,
    mime,
    suffix,
)


class TextConverter:
    name = "text"
    priority = 10

    def matches(self, attachment, data, context) -> int:
        if html_like(data, attachment):
            return 0
        if suffix(attachment) in TEXT_EXTENSIONS:
            return 30
        if looks_like_text(data):
            return 12
        return 0

    def convert(self, attachment, data, context) -> Document:
        text = decode_text(data)
        if text is None:
            raise ValueError("file is not valid UTF-8/UTF-16/Latin-1 text")
        suffix_name = suffix(attachment)
        document = Document(source_format="text")
        if suffix_name == ".json":
            try:
                text = json.dumps(
                    json.loads(text), indent=2, ensure_ascii=False
                )
            except ValueError:
                pass
        if suffix_name == ".jsonl":
            text = _pretty_jsonl(text)
        language = LANGUAGE_BY_SUFFIX.get(suffix_name, "text")
        document.code(text, language)
        return document


def _pretty_jsonl(text: str) -> str:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.dumps(json.loads(line), ensure_ascii=False))
        except ValueError:
            return text
    return "\n".join(rows)


class CsvConverter:
    name = "csv"
    priority = 40

    def matches(self, attachment, data, context) -> int:
        if suffix(attachment) not in {".csv", ".tsv"}:
            return 0
        return 60

    def convert(self, attachment, data, context) -> Document:
        text = decode_text(data) or ""
        delimiter = "\t" if suffix(attachment) == ".tsv" else ","
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        except csv.Error:
            rows = []
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        document = Document(source_format="csv" if delimiter == "," else "tsv")
        limits = context.limits
        if not rows:
            document.note("The table is empty.")
            return document
        width = max(len(row) for row in rows)
        if len(rows) > limits.max_table_rows or width > limits.max_table_columns:
            document.code(text, "csv")
            document.warn(
                f"Table has {len(rows)} rows × {width} columns; showing raw CSV "
                f"instead of a {limits.max_table_rows}×{limits.max_table_columns} table."
            )
            return document
        document.table(rows)
        return document


class _HtmlBlocks(HTMLParser):
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "section", "article", "header", "footer", "tr"}
    )
    _SKIP: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "noscript", "template", "svg"}
    )
    _HEADINGS: ClassVar[dict[str, int]] = {
        "h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, object]] = []
        self._skip = 0
        self._mode: str | None = None
        self._level = 2
        self._buffer: list[str] = []
        self._items: list[str] = []
        self._in_list = False
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._pre: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in self._HEADINGS:
            self._flush_text()
            self._mode = "heading"
            self._level = self._HEADINGS[tag]
        elif tag in {"ul", "ol"}:
            self._flush_text()
            self._items = []
            self._in_list = True
        elif tag == "li":
            self._flush_text()
            self._mode = "list-item"
        elif tag == "table":
            self._flush_text()
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "pre":
            self._flush_text()
            self._pre = []
        elif tag == "br":
            self._append("\n")
        elif tag in self._BLOCK and self._mode != "list-item":
            self._flush_text()
            self._mode = "paragraph"

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if tag in self._HEADINGS:
            self._flush_text()
        elif tag in {"ul", "ol"}:
            self._flush_text()
            if self._items:
                self.blocks.append(("list", list(self._items)))
            self._items = []
            self._in_list = False
        elif tag == "li":
            self._flush_text()
        elif tag in {"td", "th"}:
            if self._cell is not None and self._row is not None:
                self._row.append(_clean("".join(self._cell)))
                self._cell = None
        elif tag == "tr":
            if (
                self._row is not None
                and self._table is not None
                and any(self._row)
            ):
                self._table.append(self._row)
            self._row = None
        elif tag == "table":
            self._flush_text()
            if self._table:
                self.blocks.append(("table", self._table))
            self._table = None
        elif tag == "pre":
            if self._pre is not None:
                text = "".join(self._pre).rstrip("\n")
                if text:
                    self.blocks.append(("code", text))
            self._pre = None
        elif tag in self._BLOCK:
            self._flush_text()

    def handle_data(self, data):
        if self._skip:
            return
        if self._pre is not None:
            self._pre.append(data)
            return
        self._append(data)

    def _append(self, text: str) -> None:
        if self._cell is not None:
            self._cell.append(text)
        elif self._mode in {"heading", "paragraph", "list-item"}:
            self._buffer.append(text)

    def _flush_text(self) -> None:
        if self._mode == "heading":
            text = _clean("".join(self._buffer))
            if text:
                self.blocks.append(("heading", (self._level, text)))
        elif self._mode == "paragraph":
            text = _clean("".join(self._buffer))
            if text:
                self.blocks.append(("paragraph", text))
        elif self._mode == "list-item":
            text = _clean("".join(self._buffer))
            if text:
                if self._in_list:
                    self._items.append(text)
                else:
                    self.blocks.append(("paragraph", text))
        self._mode = None
        self._buffer = []

    def finish(self) -> None:
        self._flush_text()
        if self._items:
            self.blocks.append(("list", list(self._items)))
            self._items = []


def _clean(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value).strip()


class HtmlConverter:
    name = "html"
    priority = 50

    def matches(self, attachment, data, context) -> int:
        if html_like(data, attachment):
            return 70
        if mime(attachment) in {"text/html", "application/xhtml+xml"}:
            return 60
        return 0

    def convert(self, attachment, data, context) -> Document:
        text = decode_text(data) or data.decode("utf-8", errors="replace")
        return html_to_document(text)


def html_to_document(text: str) -> Document:
    parser = _HtmlBlocks()
    document = Document(source_format="html")
    try:
        parser.feed(text)
        parser.finish()
    except Exception:  # noqa: BLE001 - malformed HTML is common
        document.paragraph(text)
        document.warn("HTML parsing failed; showing the raw source.")
        return document
    for kind, payload in parser.blocks:
        if kind == "heading":
            level, title = payload  # type: ignore[misc]
            document.heading(str(title), int(level))
        elif kind == "paragraph":
            document.paragraph(str(payload))
        elif kind == "list":
            document.items([str(item) for item in payload])  # type: ignore[union-attr]
        elif kind == "table":
            document.table([list(row) for row in payload])  # type: ignore[union-attr]
        elif kind == "code":
            document.code(str(payload), "text")
    return document
