"""Converter registry and shared conversion limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .model import Document


@dataclass
class ConversionLimits:
    max_bytes: int = 20 * 1024 * 1024
    max_chars: int = 20_000
    max_table_rows: int = 50
    max_table_columns: int = 12
    max_pdf_pages: int = 50
    max_archive_entries: int = 200
    max_archive_depth: int = 1
    max_archive_total_bytes: int = 20 * 1024 * 1024


@dataclass
class ConversionContext:
    """Everything an adapter may need beyond the raw bytes."""

    limits: ConversionLimits
    registry: Any = None
    depth: int = 0


@runtime_checkable
class Converter(Protocol):
    name: str
    priority: int

    def matches(self, attachment: Any, data: bytes, context: ConversionContext) -> int:
        ...

    def convert(
        self, attachment: Any, data: bytes, context: ConversionContext
    ) -> Document:
        ...


class ConverterRegistry:
    def __init__(self) -> None:
        self._converters: list[Converter] = []

    def register(self, converter: Converter) -> None:
        if any(item.name == converter.name for item in self._converters):
            return
        self._converters.append(converter)
        self._converters.sort(key=lambda item: item.priority, reverse=True)

    def converters(self) -> tuple[Converter, ...]:
        return tuple(self._converters)

    def convert(
        self,
        attachment: Any,
        data: bytes,
        limits: ConversionLimits | None = None,
        *,
        context: ConversionContext | None = None,
    ) -> Document:
        if context is None:
            context = ConversionContext(limits or ConversionLimits(), registry=self)
        elif context.registry is None:
            context.registry = self
        best: tuple[int, Converter] | None = None
        for converter in self._converters:
            try:
                score = int(converter.matches(attachment, data, context))
            except Exception:  # noqa: BLE001 - a broken adapter must not abort discovery
                score = 0
            if score > 0 and (best is None or score > best[0]):
                best = (score, converter)
        if best is None:
            raise ValueError("no converter matched this file")
        return best[1].convert(attachment, data, context)

    def capabilities(self) -> list[str]:
        return [converter.name for converter in self._converters]


def default_registry() -> ConverterRegistry:
    """Build the standard converter set.

    Imports are local so optional adapters (pypdf) stay lazy and a broken
    optional dependency cannot prevent the core from importing.
    """

    from .adapters.archive import ArchiveConverter
    from .adapters.binary import BinaryConverter
    from .adapters.epub import EpubConverter
    from .adapters.legacy import LegacyOfficeConverter
    from .adapters.mail import EmailConverter
    from .adapters.odf import OdfConverter
    from .adapters.office import OfficeConverter
    from .adapters.pdf import PdfConverter
    from .adapters.rtf import RtfConverter
    from .adapters.text import CsvConverter, HtmlConverter, TextConverter

    registry = ConverterRegistry()
    for converter in (
        OfficeConverter(),
        OdfConverter(),
        EpubConverter(),
        PdfConverter(),
        RtfConverter(),
        LegacyOfficeConverter(),
        EmailConverter(),
        HtmlConverter(),
        CsvConverter(),
        TextConverter(),
        ArchiveConverter(),
        BinaryConverter(),
    ):
        registry.register(converter)
    return registry
