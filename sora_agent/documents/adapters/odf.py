"""ODF (ODT/ODS/ODP) adapters, using the ZIP + XML structure directly."""

from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree

from ..model import Document
from ..registry import ConversionContext
from ..sniff import read_zip_member, zip_names

OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"

_MIME_PREFIX = "application/vnd.oasis.opendocument."


class OdfConverter:
    name = "odf"
    priority = 75

    def matches(self, attachment, data, context) -> int:
        names = zip_names(data)
        return 90 if "content.xml" in names and "mimetype" in names else 0

    def convert(self, attachment, data, context) -> Document:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            mimetype = ""
            if "mimetype" in bundle.namelist():
                mimetype = bundle.read("mimetype").decode("ascii", "ignore").strip()
            root = ElementTree.fromstring(
                read_zip_member(bundle, "content.xml", max_bytes=context.limits.max_bytes)
            )
        body = root.find(f".//{OFFICE}body")
        if body is None:
            return _empty("ODF")
        if mimetype.endswith("spreadsheet") or body.find(OFFICE + "spreadsheet") is not None:
            return _spreadsheet(body, context)
        if mimetype.endswith("presentation") or body.find(OFFICE + "presentation") is not None:
            return _presentation(body)
        return _text_document(body)


def _empty(kind: str) -> Document:
    document = Document(source_format=kind.lower())
    document.note("The document contains no convertible text.")
    return document


def _itertext(node) -> str:
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _text_document(body) -> Document:
    document = Document(source_format="odt")
    text = body.find(OFFICE + "text")
    if text is None:
        return _empty("ODT")
    for child in text:
        if child.tag == TEXT + "h":
            level = int(child.attrib.get(TEXT + "outline-level", "2") or 2)
            value = _itertext(child)
            if value:
                document.heading(value, max(1, min(6, level)))
        elif child.tag == TEXT + "p":
            value = _itertext(child)
            if value:
                document.paragraph(value)
        elif child.tag == TEXT + "list":
            items = [
                _itertext(item)
                for item in child.iter(TEXT + "list-item")
                if _itertext(item)
            ]
            if items:
                document.items(items)
        elif child.tag == TABLE + "table":
            rows = _table_rows(child)
            if rows:
                document.table(rows)
    return document


def _table_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.iter(TABLE + "table-row"):
        cells: list[str] = []
        for cell in row:
            if cell.tag not in {TABLE + "table-cell", TABLE + "covered-table-cell"}:
                continue
            repeats = int(cell.attrib.get(TABLE + "number-columns-repeated", "1") or 1)
            value = (
                cell.attrib.get(OFFICE + "value")
                or cell.attrib.get(OFFICE + "string-value")
                or _itertext(cell)
            )
            value = str(value).strip()
            cells.extend([value] * min(max(1, repeats), 50))
        if any(cells):
            rows.append(cells)
    return rows


def _spreadsheet(body, context: ConversionContext) -> Document:
    document = Document(source_format="ods")
    spreadsheet = body.find(OFFICE + "spreadsheet")
    if spreadsheet is None:
        return _empty("ODS")
    for sheet in spreadsheet.findall(TABLE + "table"):
        name = sheet.attrib.get(TABLE + "name", "Sheet")
        rows = _table_rows(sheet)
        if not rows:
            continue
        document.heading(name, 3)
        width = max(len(row) for row in rows)
        rows = [row[: context.limits.max_table_columns] for row in rows[: context.limits.max_table_rows]]
        document.table(rows, html=width > context.limits.max_table_columns)
    return document


def _presentation(body) -> Document:
    document = Document(source_format="odp")
    presentation = body.find(OFFICE + "presentation")
    if presentation is None:
        return _empty("ODP")
    for index, page in enumerate(presentation.findall(DRAW + "page"), 1):
        name = page.attrib.get(DRAW + "name", f"Slide {index}")
        document.heading(f"Slide {index}: {name}".strip(), 3)
        for paragraph in page.iter(TEXT + "p"):
            value = _itertext(paragraph)
            if value:
                document.paragraph(value)
    return document
