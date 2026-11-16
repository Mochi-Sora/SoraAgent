"""Fallback for files that no structured adapter can read."""

from __future__ import annotations

import hashlib

from ..model import Document

_PREVIEW_BYTES = 256


class BinaryConverter:
    name = "binary"
    priority = 0

    def matches(self, attachment, data, context) -> int:
        return 1

    def convert(self, attachment, data, context) -> Document:
        document = Document(source_format="binary")
        digest = hashlib.sha256(data).hexdigest()
        preview = data[:_PREVIEW_BYTES].hex(" ")
        document.paragraph("**Binary file metadata**")
        document.items(
            [
                f"SHA-256: `{digest}`",
                f"Size: {_format_size(len(data))}",
            ]
        )
        document.heading("First bytes (hex)", 3)
        document.code(preview, "text")
        document.note("No text representation is available for this file type.")
        return document


def _format_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
