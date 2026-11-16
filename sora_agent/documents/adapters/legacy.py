"""Detect legacy OLE Office files and return an actionable capability gap."""

from __future__ import annotations

from ..model import Document
from ..sniff import is_ole


class LegacyOfficeConverter:
    name = "legacy-office"
    priority = 70

    def matches(self, attachment, data, context) -> int:
        if is_ole(data):
            return 75
        return 0

    def convert(self, attachment, data, context) -> Document:
        document = Document(source_format="legacy-office")
        document.paragraph("**Legacy binary Office file**")
        document.note(
            "This is an OLE binary .doc/.xls/.ppt file. The core converter "
            "does not read that container. Re-save it as DOCX/XLSX/PPTX, or "
            "configure an external converter (LibreOffice headless, antiword "
            "for .doc, or catdoc/catppt) in a later adapter."
        )
        return document
