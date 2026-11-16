"""RFC 822 email (.eml) adapter."""

from __future__ import annotations

import email
from email import policy

from ..model import Document
from ..sniff import looks_like_text, suffix
from .text import html_to_document


class EmailConverter:
    name = "email"
    priority = 55

    def matches(self, attachment, data, context) -> int:
        if suffix(attachment) not in {".eml", ".msg"}:
            return 0
        if suffix(attachment) == ".msg":
            return 0
        if not looks_like_text(data):
            return 0
        head = data[:200].lower()
        return 65 if b"from:" in head or b"mime-version:" in head else 0

    def convert(self, attachment, data, context) -> Document:
        document = Document(source_format="eml")
        message = email.message_from_bytes(data, policy=policy.default)
        rows = [
            ["Header", "Value"],
            ["From", str(message.get("From", ""))],
            ["To", str(message.get("To", ""))],
            ["Cc", str(message.get("Cc", ""))],
            ["Date", str(message.get("Date", ""))],
            ["Subject", str(message.get("Subject", ""))],
        ]
        document.table([row for row in rows if row[1] or row[0] == "Header"])
        body_text = ""
        html_body = ""
        attachments: list[str] = []
        for part in message.walk():
            if part.is_multipart():
                continue
            disposition = str(part.get_content_disposition() or "")
            filename = part.get_filename()
            if disposition == "attachment" or filename:
                attachments.append(filename or part.get_content_type())
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain" and not body_text:
                try:
                    body_text = part.get_content()
                except (LookupError, ValueError):
                    body_text = ""
            elif content_type == "text/html" and not html_body:
                try:
                    html_body = part.get_content()
                except (LookupError, ValueError):
                    html_body = ""
        if body_text and body_text.strip():
            document.heading("Body", 3)
            document.paragraph(body_text.strip())
        elif html_body:
            chapter = html_to_document(html_body)
            document.heading("Body", 3)
            document.blocks.extend(chapter.blocks)
        if attachments:
            document.heading("Attachments", 3)
            document.items(attachments)
        return document
