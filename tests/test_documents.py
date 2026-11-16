"""Tests for the document IR and the new format adapters."""

from __future__ import annotations

import io
import zipfile

from sora_agent.documents import (
    ConversionLimits,
    default_registry,
    render_markdown,
)
from sora_agent.files import Attachment, FileUnderstander

OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
DRAW = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"


def _zip(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return buffer.getvalue()


def _odf(body: str, mimetype: str) -> bytes:
    content = (
        f'<office:document-content xmlns:office="{OFFICE}" '
        f'xmlns:text="{TEXT}" xmlns:table="{TABLE}" '
        f'xmlns:draw="{DRAW}" office:version="1.3">'
        f"<office:body>{body}</office:body></office:document-content>"
    )
    return _zip({"mimetype": mimetype, "content.xml": content})


def test_odt_headings_lists_and_tables():
    body = (
        "<office:text>"
        '<text:h text:outline-level="1">ODT title</text:h>'
        "<text:p>Hello ODT</text:p>"
        "<text:list><text:list-item><text:p>One</text:p></text:list-item>"
        "<text:list-item><text:p>Two</text:p></text:list-item></text:list>"
        "<table:table><table:table-row><table:table-cell><text:p>Name</text:p>"
        "</table:table-cell><table:table-cell><text:p>Ada</text:p>"
        "</table:table-cell></table:table-row></table:table>"
        "</office:text>"
    )
    data = _odf(body, "application/vnd.oasis.opendocument.text")
    markdown = FileUnderstander().understand([Attachment("notes.odt", data=data)])

    assert "ODT title" in markdown
    assert "Hello ODT" in markdown
    assert "- One" in markdown
    assert "| Name | Ada |" in markdown


def test_ods_sheet_becomes_a_table():
    body = (
        "<office:spreadsheet><table:table table:name=\"Budget\">"
        '<table:table-row><table:table-cell office:value-type="string">'
        "<text:p>Item</text:p></table:table-cell>"
        '<table:table-cell office:value-type="string"><text:p>Cost</text:p>'
        "</table:table-cell></table:table-row>"
        '<table:table-row><table:table-cell><text:p>Hosting</text:p>'
        '</table:table-cell><table:table-cell office:value="10">'
        "<text:p>10</text:p></table:table-cell></table:table-row>"
        "</table:table></office:spreadsheet>"
    )
    data = _odf(body, "application/vnd.oasis.opendocument.spreadsheet")
    markdown = FileUnderstander().understand([Attachment("budget.ods", data=data)])

    assert "Budget" in markdown
    assert "| Item | Cost |" in markdown
    assert "| Hosting | 10 |" in markdown


def test_odp_slides_are_rendered():
    body = (
        "<office:presentation>"
        '<draw:page draw:name="Intro"><draw:frame><draw:text-box>'
        "<text:p>Welcome aboard</text:p></draw:text-box></draw:frame>"
        "</draw:page></office:presentation>"
    )
    data = _odf(body, "application/vnd.oasis.opendocument.presentation")
    markdown = FileUnderstander().understand([Attachment("deck.odp", data=data)])

    assert "Slide 1" in markdown
    assert "Welcome aboard" in markdown


def test_epub_spine_is_read():
    opf = (
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Test Book</dc:title></metadata>"
        '<manifest><item id="c1" href="chapter1.xhtml" '
        'media-type="application/xhtml+xml"/></manifest>'
        "<spine><itemref idref=\"c1\"/></spine></package>"
    )
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles>'
        "</container>"
    )
    chapter = (
        "<html><body><h1>Chapter One</h1><p>Once upon a time.</p></body></html>"
    )
    data = _zip(
        {
            "mimetype": "application/epub+zip",
            "META-INF/container.xml": container,
            "OEBPS/content.opf": opf,
            "OEBPS/chapter1.xhtml": chapter,
        }
    )
    markdown = FileUnderstander().understand([Attachment("book.epub", data=data)])

    assert "Test Book" in markdown
    assert "Chapter One" in markdown
    assert "Once upon a time." in markdown


def test_eml_headers_and_body():
    data = (
        b"From: Ada <ada@example.com>\n"
        b"To: Grace <grace@example.com>\n"
        b"Subject: Hello\n"
        b"Date: Mon, 1 Jan 2024 10:00:00 +0000\n"
        b"MIME-Version: 1.0\n"
        b"Content-Type: text/plain; charset=utf-8\n\n"
        b"Body of the email.\n"
    )
    markdown = FileUnderstander().understand([Attachment("mail.eml", data=data)])

    assert "ada@example.com" in markdown
    assert "Hello" in markdown
    assert "Body of the email." in markdown


def test_nested_archive_extracts_inner_text():
    inner = _zip({"note.txt": "nested hello"})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("inner.zip", inner)
    outer = buffer.getvalue()

    markdown = FileUnderstander().understand([Attachment("outer.zip", data=outer)])

    assert "inner.zip" in markdown
    assert "nested hello" in markdown


def test_docx_tables_and_headings_keep_structure():
    document = _zip(
        {
            "word/document.xml": (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:body>'
                '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                "<w:r><w:t>Report</w:t></w:r></w:p>"
                "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>"
                "<w:tc><w:p><w:r><w:t>Score</w:t></w:r></w:p></w:tc></w:tr>"
                "<w:tr><w:tc><w:p><w:r><w:t>Ada</w:t></w:r></w:p></w:tc>"
                "<w:tc><w:p><w:r><w:t>10</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
                "</w:body></w:document>"
            )
        }
    )
    markdown = FileUnderstander().understand([Attachment("report.docx", data=document)])

    assert "# Report" in markdown
    assert "| Name | Score |" in markdown
    assert "| Ada | 10 |" in markdown


def test_registry_reports_converters_and_limits():
    registry = default_registry()
    assert {"ooxml", "odf", "epub", "pdf", "text", "csv", "html", "archive", "binary"} <= set(
        registry.capabilities()
    )
    document = registry.convert(
        Attachment("plain.txt", data=b"hello"),
        b"hello",
        ConversionLimits(max_chars=100),
    )
    assert "hello" in render_markdown(document)


def test_rtf_extracts_text_and_skips_font_table():
    data = (
        r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Helvetica;}}"
        r"\f0 Hello from RTF.\par Second paragraph.\par}"
    ).encode("latin-1")
    markdown = FileUnderstander().understand([Attachment("note.rtf", data=data)])

    assert "Hello from RTF." in markdown
    assert "Second paragraph." in markdown
    assert "Helvetica" not in markdown


def test_legacy_office_reports_a_capability_gap():
    data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32
    markdown = FileUnderstander().understand(
        [Attachment("old.doc", data=data)]
    )

    assert "Legacy binary Office file" in markdown
    assert "DOCX/XLSX/PPTX" in markdown


def test_rtf_backslash_newline_is_a_line_break():
    data = b"{\\rtf1 First\\\n\\\nSecond\\\n- item}"
    markdown = FileUnderstander().understand([Attachment("line.rtf", data=data)])

    assert "First" in markdown
    assert "Second" in markdown
    assert "- item" in markdown
    assert "FirstSecond" not in markdown


def test_html_lists_and_tables_are_structured():
    data = (
        b"<html><body><h1>Doc</h1><ul><li>One</li><li>Two</li></ul>"
        b"<table><tr><th>Name</th><th>Score</th></tr>"
        b"<tr><td>Ada</td><td>10</td></tr></table></body></html>"
    )
    markdown = FileUnderstander().understand([Attachment("page.html", data=data)])

    assert "# Doc" in markdown
    assert "- One" in markdown and "- Two" in markdown
    assert "| Name | Score |" in markdown
    assert "| Ada | 10 |" in markdown


def test_xlsx_uses_sheet_names_and_formulas():
    workbook = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Budget" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="inlineStr"><is><t>Item</t></is></c>'
        '<c r="B1" t="inlineStr"><is><t>Cost</t></is></c></row>'
        '<row r="2"><c r="A2" t="inlineStr"><is><t>Hosting</t></is></c>'
        '<c r="B2"><v>10</v></c></row>'
        '<row r="3"><c r="A3" t="inlineStr"><is><t>Total</t></is></c>'
        '<c r="B3"><f>SUM(B2:B2)</f></c></row>'
        "</sheetData></worksheet>"
    )
    data = _zip(
        {
            "xl/workbook.xml": workbook,
            "xl/_rels/workbook.xml.rels": rels,
            "xl/worksheets/sheet1.xml": sheet,
        }
    )
    markdown = FileUnderstander().understand([Attachment("budget.xlsx", data=data)])

    assert "Budget" in markdown
    assert "| Item | Cost |" in markdown
    assert "| Hosting | 10 |" in markdown
    assert "=SUM(B2:B2)" in markdown


def test_archive_total_byte_cap_reports_truncation():
    big = "x" * 2_000_000
    data = _zip({"big.txt": big})
    limits = ConversionLimits(max_archive_total_bytes=100_000)
    document = default_registry().convert(
        Attachment("big.zip", data=data), data, limits
    )
    rendered = render_markdown(document)
    assert "total-byte" in rendered
    assert big not in rendered
