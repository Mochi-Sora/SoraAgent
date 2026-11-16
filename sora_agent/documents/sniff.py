"""Format sniffing and small container helpers for document adapters."""

from __future__ import annotations

import mimetypes
import re
import zipfile
from pathlib import Path
from typing import Any

TEXT_EXTENSIONS = frozenset(
    {
        ".c", ".cfg", ".conf", ".cpp", ".css", ".csv", ".go", ".h", ".hpp",
        ".ini", ".java", ".js", ".json", ".jsonl", ".jsx", ".log", ".md",
        ".php", ".ps1", ".py", ".rb", ".rs", ".sh", ".sql", ".svg", ".toml",
        ".ts", ".tsv", ".tsx", ".txt", ".xml", ".yaml", ".yml", ".ics",
    }
)

LANGUAGE_BY_SUFFIX = {
    ".c": "c", ".cfg": "ini", ".conf": "ini", ".cpp": "cpp", ".css": "css",
    ".csv": "csv", ".go": "go", ".h": "c", ".hpp": "cpp", ".ini": "ini",
    ".java": "java", ".js": "javascript", ".json": "json", ".jsonl": "json",
    ".jsx": "jsx", ".log": "text", ".md": "markdown", ".php": "php",
    ".ps1": "powershell", ".py": "python", ".rb": "ruby", ".rs": "rust",
    ".sh": "bash", ".sql": "sql", ".svg": "xml", ".toml": "toml",
    ".ts": "typescript", ".tsx": "tsx", ".txt": "text", ".xml": "xml",
    ".yaml": "yaml", ".yml": "yaml", ".ics": "text",
}

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)

OFFICE_EXTENSIONS = frozenset(
    {".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp", ".epub"}
)

ARCHIVE_EXTENSIONS = frozenset({".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"})


def suffix(attachment: Any) -> str:
    name = str(getattr(attachment, "filename", "") or "")
    try:
        return Path(name).suffix.lower()
    except (OSError, ValueError):
        return ""


def mime(attachment: Any) -> str:
    value = str(getattr(attachment, "mime_type", "") or "").lower()
    if value:
        return value
    name = str(getattr(attachment, "filename", "") or "")
    return (mimetypes.guess_type(name)[0] or "").lower()


def looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if not sample:
        return True
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
    printable = sum(
        1 for byte in sample if byte in b"\t\n\r" or 32 <= byte < 127
    )
    return printable / len(sample) >= 0.85


def decode_text(data: bytes) -> str | None:
    if not looks_like_text(data):
        return None
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def is_zip(data: bytes) -> bool:
    return data.startswith((b"PK\x03\x04", b"PK\x05\x06"))


def is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF-")


def is_ole(data: bytes) -> bool:
    return data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")


def is_rtf(data: bytes) -> bool:
    return data.lstrip().startswith(b"{\\rtf")


def zip_names(data: bytes) -> set[str]:
    try:
        with zipfile.ZipFile(_bytes_io(data)) as bundle:
            return {name.replace("\\", "/") for name in bundle.namelist()}
    except (OSError, zipfile.BadZipFile):
        return set()


def _bytes_io(data: bytes):
    import io

    return io.BytesIO(data)


def read_zip_member(bundle: zipfile.ZipFile, name: str, *, max_bytes: int) -> bytes:
    info = bundle.getinfo(name)
    if info.file_size > max_bytes:
        raise ValueError(f"{name} is too large to read safely ({info.file_size} bytes)")
    return bundle.read(name)


def html_like(data: bytes, attachment: Any) -> bool:
    if suffix(attachment) in {".html", ".htm", ".xhtml"}:
        return True
    if mime(attachment) in {"text/html", "application/xhtml+xml"}:
        return True
    head = data[:2048].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html"))


def first_xml_text(root: Any, endings: tuple[str, ...]) -> list[str]:
    return [
        node.text or ""
        for node in root.iter()
        if any(node.tag.endswith(ending) for ending in endings) and node.text
    ]


def inline(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
