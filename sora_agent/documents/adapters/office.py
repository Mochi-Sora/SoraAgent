"""DOCX, PPTX, and XLSX adapters.

These are standard-library readers: OOXML is a ZIP of XML parts.  The goal is
structure (headings, lists, tables, notes, sheet names) rather than a raw text
dump.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from xml.etree import ElementTree

from ..model import Document
from ..registry import ConversionContext
from ..sniff import read_zip_member, zip_names

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"

_HEADING_RE = re.compile(r"^(?:heading|title)\s*([1-9])?$", re.IGNORECASE)


class OfficeConverter:
    name = "ooxml"
    priority = 80

    def matches(self, attachment, data, context) -> int:
        names = zip_names(data)
        if not names:
            return 0
        if "word/document.xml" in names:
            return 100
        if any(name.startswith("ppt/slides/slide") and name.endswith(".xml") for name in names):
            return 100
        if "xl/workbook.xml" in names or any(
            name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            for name in names
        ):
            return 100
        return 0

    def convert(self, attachment, data, context) -> Document:
        names = zip_names(data)
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            if "word/document.xml" in names:
                return _docx(bundle, context)
            if "xl/workbook.xml" in names or any(
                name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                for name in names
            ):
                return _xlsx(bundle, context)
            return _pptx(bundle, context)


def _text_of(node, namespace: str = W, tag: str = "t") -> str:
    return "".join(
        child.text or ""
        for child in node.iter()
        if child.tag == namespace + tag and child.text
    )


def _docx(bundle: zipfile.ZipFile, context: ConversionContext) -> Document:
    document = Document(source_format="docx")
    raw = read_zip_member(
        bundle, "word/document.xml", max_bytes=context.limits.max_bytes
    )
    root = ElementTree.fromstring(raw)
    body = root.find(W + "body")
    if body is None:
        document.warn("The DOCX has no document body.")
        return document
    pending_list: list[str] = []
    images = 0

    def flush_list() -> None:
        nonlocal pending_list
        if pending_list:
            document.items(pending_list)
            pending_list = []

    for child in body:
        if child.tag == W + "p":
            style_node = child.find(f"{W}pPr/{W}pStyle")
            style = ""
            if style_node is not None:
                style = style_node.attrib.get(W + "val", "")
            text = _text_of(child).strip()
            if not text:
                continue
            heading = _HEADING_RE.match(style)
            if heading:
                flush_list()
                level = int(heading.group(1) or 1)
                document.heading(text, level)
            elif child.find(f"{W}pPr/{W}numPr") is not None:
                pending_list.append(text)
            else:
                flush_list()
                document.paragraph(text)
            images += len(child.findall(f".//{W}drawing"))
        elif child.tag == W + "tbl":
            flush_list()
            rows, merged = _docx_table(child)
            if rows:
                document.table(rows, html=merged)
        elif child.tag == W + "sdt":
            # Content controls wrap paragraphs/tables; flatten the text path.
            text = _text_of(child).strip()
            if text:
                flush_list()
                document.paragraph(text)
        if len(document.blocks) >= 400:
            document.warn("The DOCX is large; only the first part was converted.")
            break
    flush_list()
    if images:
        document.note(f"{images} embedded image(s) were not extracted.")
    return document


def _docx_table(table) -> tuple[list[list[str]], bool]:
    rows: list[list[str]] = []
    merged = False
    for tr in table.findall(W + "tr"):
        row: list[str] = []
        for tc in tr.findall(W + "tc"):
            span = tc.find(f"{W}tcPr/{W}gridSpan")
            if span is not None:
                merged = True
            value = " ".join(
                _text_of(paragraph).strip()
                for paragraph in tc.findall(W + "p")
            ).strip()
            row.append(value)
        if any(row):
            rows.append(row)
    return rows, merged


def _pptx(bundle: zipfile.ZipFile, context: ConversionContext) -> Document:
    document = Document(source_format="pptx")
    slides = sorted(
        (
            name
            for name in bundle.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ),
        key=lambda name: int(re.search(r"(\d+)", name).group(1)),  # type: ignore[union-attr]
    )
    for index, name in enumerate(slides[:120], 1):
        root = ElementTree.fromstring(
            read_zip_member(bundle, name, max_bytes=context.limits.max_bytes)
        )
        document.heading(f"Slide {index}", 3)
        tree = root.find(f".//{P}spTree")
        if tree is None:
            for paragraph in root.findall(f".//{A}p"):
                text = "".join(
                    node.text or "" for node in paragraph.iter(A + "t") if node.text
                ).strip()
                if text:
                    document.paragraph(text)
            continue
        for shape in list(tree):
            if shape.tag == P + "sp":
                for paragraph in shape.findall(f".//{A}p"):
                    text = "".join(
                        node.text or ""
                        for node in paragraph.iter(A + "t")
                        if node.text
                    ).strip()
                    if text:
                        document.paragraph(text)
            elif shape.tag == P + "graphicFrame":
                for table in shape.findall(f".//{A}tbl"):
                    rows = []
                    for tr in table.findall(A + "tr"):
                        row = [
                            " ".join(
                                node.text or ""
                                for node in cell.iter(A + "t")
                                if node.text
                            ).strip()
                            for cell in tr.findall(A + "tc")
                        ]
                        if any(row):
                            rows.append(row)
                    if rows:
                        document.table(rows)
            elif shape.tag == P + "pic":
                document.note("Embedded image (not extracted).")
        note_name = f"ppt/notesSlides/notesSlide{index}.xml"
        if note_name in bundle.namelist():
            note_root = ElementTree.fromstring(
                read_zip_member(bundle, note_name, max_bytes=context.limits.max_bytes)
            )
            note = " ".join(
                node.text or ""
                for node in note_root.iter(A + "t")
                if node.text
            ).strip()
            if note:
                document.note("Speaker notes: " + note)
    if len(slides) > 120:
        document.warn("Only the first 120 slides were converted.")
    return document


def _xlsx(bundle: zipfile.ZipFile, context: ConversionContext) -> Document:
    document = Document(source_format="xlsx")
    shared = _shared_strings(bundle, context)
    sheets = _sheet_parts(bundle, context)
    if not sheets:
        sheets = [
            (f"Sheet {index}", name, False)
            for index, name in enumerate(
                sorted(
                    item
                    for item in bundle.namelist()
                    if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", item)
                ),
                1,
            )
        ]
    for name, path, hidden in sheets[:50]:
        if path not in bundle.namelist():
            continue
        root = ElementTree.fromstring(
            read_zip_member(bundle, path, max_bytes=context.limits.max_bytes)
        )
        rows, row_truncated, column_truncated = _sheet_rows(
            root,
            shared,
            context.limits.max_table_rows,
            context.limits.max_table_columns,
        )
        if not rows:
            continue
        document.heading(name + (" (hidden)" if hidden else ""), 3)
        merged = root.find(f".//{S}mergeCells") is not None
        document.table(rows, html=merged)
        if merged:
            document.warn(f"Sheet {name!r} has merged cells; rendered as HTML.")
        if row_truncated or column_truncated:
            document.warn(
                f"Sheet {name!r} was truncated to "
                f"{context.limits.max_table_rows} rows × "
                f"{context.limits.max_table_columns} columns."
            )
    if len(sheets) > 50:
        document.warn("Only the first 50 sheets were converted.")
    return document


def _shared_strings(bundle: zipfile.ZipFile, context: ConversionContext) -> list[str]:
    if "xl/sharedStrings.xml" not in bundle.namelist():
        return []
    root = ElementTree.fromstring(
        read_zip_member(bundle, "xl/sharedStrings.xml", max_bytes=context.limits.max_bytes)
    )
    values = []
    for item in root.iter(S + "si"):
        values.append("".join(node.text or "" for node in item.iter(S + "t")))
    return values


def _sheet_parts(bundle: zipfile.ZipFile, context: ConversionContext):
    try:
        workbook = ElementTree.fromstring(
            read_zip_member(bundle, "xl/workbook.xml", max_bytes=context.limits.max_bytes)
        )
    except (KeyError, ElementTree.ParseError):
        return []
    targets: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in bundle.namelist():
        rels = ElementTree.fromstring(
            read_zip_member(bundle, "xl/_rels/workbook.xml.rels", max_bytes=context.limits.max_bytes)
        )
        for rel in rels:
            if rel.tag.endswith("}Relationship"):
                target = rel.attrib.get("Target", "").strip()
                if target.startswith("/"):
                    target = target[1:]
                else:
                    target = posixpath.normpath(posixpath.join("xl", target))
                targets[rel.attrib.get("Id", "")] = target
    parts = []
    for sheet in workbook.iter(S + "sheet"):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(R + "id", "")
        path = targets.get(rel_id, "")
        hidden = sheet.attrib.get("state", "") == "hidden"
        if path:
            parts.append((name, path, hidden))
    if not parts:
        parts = [
            (f"Sheet {index}", name, False)
            for index, name in enumerate(
                sorted(
                    name
                    for name in bundle.namelist()
                    if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
                ),
                1,
            )
        ]
    return parts


def _sheet_rows(root, shared: list[str], max_rows: int, max_columns: int):
    rows = []
    row_truncated = False
    column_truncated = False
    for row in root.iter(S + "row"):
        if len(rows) >= max_rows:
            row_truncated = True
            break
        cells: dict[int, str] = {}
        for cell in row.iter(S + "c"):
            reference = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)", reference.upper())
            if not match:
                continue
            column = 0
            for char in match.group(1):
                column = column * 26 + (ord(char) - ord("A") + 1)
            column -= 1
            if column >= max_columns:
                column_truncated = True
                continue
            kind = cell.attrib.get("t", "")
            formula = cell.find(S + "f")
            value_node = cell.find(S + "v")
            value = value_node.text if value_node is not None and value_node.text else ""
            if kind == "s":
                try:
                    value = shared[int(value)]
                except (IndexError, ValueError):
                    value = ""
            elif kind == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iter(S + "t")
                )
            elif kind == "b":
                value = "TRUE" if value == "1" else "FALSE" if value else ""
            elif formula is not None and not value:
                value = "=" + (formula.text or "")
            cells[column] = value.strip()
        if cells:
            width = min(max(cells) + 1, max_columns)
            rows.append([cells.get(index, "") for index in range(width)])
    return rows, row_truncated, column_truncated
