"""Render the common document model as GitHub-flavored Markdown."""

from __future__ import annotations

import html as html_module
import re

from .model import Block, Document


def _fence(text: str, language: str = "text") -> str:
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{text.rstrip()}\n{ticks}"


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _html_table(rows: tuple[tuple[str, ...], ...], header: bool) -> str:
    width = max((len(row) for row in rows), default=0)
    parts = ["<table>"]
    for index, row in enumerate(rows):
        tag = "th" if header and index == 0 else "td"
        padded = list(row) + [""] * (width - len(row))
        cells = "".join(
            f"<{tag}>{html_module.escape(cell)}</{tag}>" for cell in padded
        )
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _markdown_table(rows: tuple[tuple[str, ...], ...], header: bool) -> str:
    width = max((len(row) for row in rows), default=0)
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    lines = []
    if header:
        lines.append("| " + " | ".join(_cell(value) for value in padded[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
        body = padded[1:]
    else:
        body = padded
    for row in body:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _render_block(block: Block) -> str:
    if block.kind == "heading":
        level = max(1, min(6, int(block.level or 2)))
        return "#" * level + " " + block.text
    if block.kind == "paragraph":
        return block.text
    if block.kind == "code":
        return _fence(block.text, block.language or "text")
    if block.kind == "quote":
        return "\n".join("> " + line for line in block.text.splitlines())
    if block.kind == "note":
        return "\n".join("> " + line for line in block.text.splitlines())
    if block.kind == "list":
        marker = "1. " if block.meta.get("ordered") else "- "
        return "\n".join(marker + item for item in block.items)
    if block.kind == "table":
        rows = block.rows
        if not rows:
            return ""
        if block.meta.get("html"):
            return _html_table(rows, bool(block.meta.get("header", True)))
        return _markdown_table(rows, bool(block.meta.get("header", True)))
    if block.kind == "image":
        alt = str(block.meta.get("alt") or "image")
        ref = str(block.meta.get("ref") or "")
        label = f"![{alt}]({ref})" if ref else f"**Image: {alt}**"
        if block.text:
            return f"{label}\n\n{block.text}"
        return label
    if block.kind == "transcript":
        return block.text
    return block.text


def render_markdown(document: Document, *, include_title: bool = False) -> str:
    """Render a document to Markdown, including conversion warnings."""

    parts: list[str] = []
    if include_title and document.title:
        parts.append(f"# {document.title}")
    for block in document.blocks:
        rendered = _render_block(block).strip()
        if rendered:
            parts.append(rendered)
    for warning in document.warnings:
        parts.append(f"> Conversion note: {warning}")
    return "\n\n".join(parts).strip()
