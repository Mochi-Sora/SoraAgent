"""Document model shared by readers and writers.

Adapters convert an input file into ``Block`` objects; the Markdown renderer
is the only place that decides how a block becomes text.  This keeps format
details out of the attachment wrapper and gives every adapter the same
provenance, table, and warning semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """One semantic block.

    ``kind`` is one of ``heading``, ``paragraph``, ``list``, ``table``,
    ``code``, ``quote``, ``note``, ``image``, or ``transcript``.
    """

    kind: str
    text: str = ""
    level: int = 0
    items: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    language: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A converted document, ready for Markdown rendering."""

    blocks: list[Block] = field(default_factory=list)
    title: str = ""
    source_format: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def heading(self, text: str, level: int = 2, **meta: Any) -> None:
        text = str(text or "").strip()
        if text:
            self.blocks.append(Block("heading", text=text, level=level, meta=meta))

    def paragraph(self, text: str, **meta: Any) -> None:
        text = str(text or "").strip()
        if text:
            self.blocks.append(Block("paragraph", text=text, meta=meta))

    def code(self, text: str, language: str = "text", **meta: Any) -> None:
        text = str(text or "").rstrip("\n")
        if text:
            self.blocks.append(
                Block("code", text=text, language=language, meta=meta)
            )

    def quote(self, text: str, **meta: Any) -> None:
        text = str(text or "").strip()
        if text:
            self.blocks.append(Block("quote", text=text, meta=meta))

    def note(self, text: str, **meta: Any) -> None:
        text = str(text or "").strip()
        if text:
            self.blocks.append(Block("note", text=text, meta=meta))

    def items(self, values: list[str], ordered: bool = False, **meta: Any) -> None:
        cleaned = tuple(str(value).strip() for value in values if str(value).strip())
        if cleaned:
            self.blocks.append(
                Block("list", items=cleaned, meta={"ordered": ordered, **meta})
            )

    def table(
        self,
        rows: list[list[str]],
        *,
        header: bool = True,
        html: bool = False,
        **meta: Any,
    ) -> None:
        cleaned = tuple(
            tuple(str(cell) for cell in row) for row in rows if row is not None
        )
        if cleaned:
            self.blocks.append(
                Block(
                    "table",
                    rows=cleaned,
                    meta={"header": header, "html": html, **meta},
                )
            )

    def image(self, alt: str, ref: str = "", description: str = "") -> None:
        self.blocks.append(
            Block(
                "image",
                text=description.strip(),
                meta={"alt": alt.strip(), "ref": ref.strip()},
            )
        )

    def warn(self, message: str) -> None:
        text = str(message or "").strip()
        if text and text not in self.warnings:
            self.warnings.append(text)
