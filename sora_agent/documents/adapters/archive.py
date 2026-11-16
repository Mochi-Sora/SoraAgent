"""ZIP/TAR archive adapter with bounded recursive conversion."""

from __future__ import annotations

import io
import tarfile
import zipfile

from ..model import Document
from ..registry import ConversionContext, ConversionLimits
from ..sniff import looks_like_text, suffix


class ArchiveConverter:
    name = "archive"
    priority = 60

    def matches(self, attachment, data, context) -> int:
        if suffix(attachment) in {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}:
            return 55
        if data.startswith(b"PK\x03\x04") and not _is_document_zip(data):
            return 45
        if data.startswith((b"\x1f\x8b", b"BZh")):
            return 45
        if data[257:262] == b"ustar":
            return 45
        return 0

    def convert(self, attachment, data, context) -> Document:
        document = Document(source_format="archive")
        limits = context.limits
        entries, content_truncated = _read_entries(data, limits)
        if not entries:
            document.note("The archive is empty.")
            return document
        document.heading("Archive contents", 3)
        document.items(
            [f"`{name}` ({_format_size(size)})" for name, size, _ in entries[: limits.max_archive_entries]]
        )
        if len(entries) > limits.max_archive_entries:
            document.warn(
                f"Only the first {limits.max_archive_entries} of {len(entries)} entries were inspected."
            )
        if content_truncated:
            document.warn(
                "Archive content extraction stopped at the total-byte or "
                "per-entry cap; remaining entries are listed only."
            )
        if context.depth >= limits.max_archive_depth:
            document.note("Nested archive expansion was stopped at the depth limit.")
            return document

        total = 0
        for name, size, payload in entries:
            if payload is None or total + len(payload) > limits.max_archive_total_bytes:
                continue
            if not looks_like_text(payload) and not _looks_like_archive(payload):
                continue
            total += len(payload)
            child = _ChildAttachment(name)
            child_limits = _child_limits(limits)
            child_context = ConversionContext(
                child_limits, registry=context.registry, depth=context.depth + 1
            )
            try:
                nested = context.registry.convert(
                    child, payload, child_limits, context=child_context
                )
            except Exception as exc:  # noqa: BLE001 - one bad member is data
                document.warn(f"{name}: {exc}")
                continue
            if not nested.blocks:
                continue
            document.heading(name, 3)
            document.blocks.extend(nested.blocks)
            document.warnings.extend(nested.warnings)
            if len(document.blocks) >= 300:
                document.warn("Archive conversion stopped after 300 blocks.")
                break
        return document


class _ChildAttachment:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.mime_type = ""
        self.kind = "document"


def _child_limits(limits: ConversionLimits) -> ConversionLimits:
    return ConversionLimits(
        max_bytes=min(limits.max_bytes, limits.max_archive_total_bytes),
        max_chars=limits.max_chars,
        max_table_rows=limits.max_table_rows,
        max_table_columns=limits.max_table_columns,
        max_pdf_pages=limits.max_pdf_pages,
        max_archive_entries=limits.max_archive_entries,
        max_archive_depth=limits.max_archive_depth,
        max_archive_total_bytes=max(0, limits.max_archive_total_bytes // 2),
    )


def _read_entries(
    data: bytes, limits: ConversionLimits
) -> tuple[list[tuple[str, int, bytes | None]], bool]:
    entries: list[tuple[str, int, bytes | None]] = []
    total = 0
    truncated = False
    try:
        if data.startswith((b"PK\x03\x04", b"PK\x05\x06")):
            with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                infos = bundle.infolist()[: limits.max_archive_entries + 1]
                for info in infos:
                    if info.is_dir():
                        continue
                    size = info.file_size
                    payload = None
                    if size <= limits.max_bytes and total + size <= limits.max_archive_total_bytes:
                        try:
                            with bundle.open(info) as handle:
                                payload = handle.read(size + 1)
                            if len(payload) > size:
                                payload = None
                                truncated = True
                            else:
                                total += len(payload)
                        except (OSError, zipfile.BadZipFile):
                            payload = None
                    else:
                        truncated = True
                    entries.append((info.filename, size, payload))
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as bundle:
                members = bundle.getmembers()[: limits.max_archive_entries + 1]
                for member in members:
                    if not member.isfile():
                        continue
                    size = member.size
                    payload = None
                    if size <= limits.max_bytes and total + size <= limits.max_archive_total_bytes:
                        try:
                            extracted = bundle.extractfile(member)
                            payload = extracted.read(size + 1) if extracted else None
                            if payload is not None and len(payload) > size:
                                payload = None
                                truncated = True
                            elif payload is not None:
                                total += len(payload)
                        except (OSError, tarfile.TarError):
                            payload = None
                    else:
                        truncated = True
                    entries.append((member.name, size, payload))
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not read this archive: {exc}") from exc
    return entries, truncated


def _is_document_zip(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            names = set(bundle.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    if "word/document.xml" in names or "xl/workbook.xml" in names or "content.xml" in names:
        return True
    return any(name.startswith("ppt/slides/slide") for name in names)


def _format_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _looks_like_archive(data: bytes) -> bool:
    return data.startswith(
        (b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00")
    ) or data[257:262] == b"ustar"
