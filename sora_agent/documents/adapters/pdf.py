"""PDF adapter using the optional pypdf package."""

from __future__ import annotations

import importlib
import io
import re

from ..model import Document


class PdfConverter:
    name = "pdf"
    priority = 85

    def matches(self, attachment, data, context) -> int:
        if data.startswith(b"%PDF-"):
            return 95
        return 0

    def convert(self, attachment, data, context) -> Document:
        document = Document(source_format="pdf")
        try:
            pypdf = importlib.import_module("pypdf")
        except ImportError:
            document.note(
                "PDF text extraction needs the optional `files` extra:\n"
                "`pip install 'sora-agent[files]'`."
            )
            return document
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = list(reader.pages[: context.limits.max_pdf_pages])
        except Exception as exc:  # noqa: BLE001 - malformed or encrypted PDF
            document.note(f"Could not read this PDF: {exc}")
            return document
        total = len(reader.pages)
        extracted = 0
        for index, page in enumerate(pages, 1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:  # noqa: BLE001 - one broken page is data
                document.warn(f"Page {index}: {exc}")
                continue
            if not text:
                continue
            extracted += 1
            document.heading(f"Page {index}", 3)
            document.code(_normalize_pdf_text(text), "text")
        if not extracted:
            document.note(
                "The PDF contained no extractable text (it may be scanned); "
                "configure an OCR backend to convert page images."
            )
        if total > context.limits.max_pdf_pages:
            document.warn(
                f"Only the first {context.limits.max_pdf_pages} of {total} pages were converted."
            )
        return document


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
