"""EPUB adapter: read the spine and reuse the HTML converter."""

from __future__ import annotations

import io
import posixpath
import zipfile
from xml.etree import ElementTree

from ..model import Document
from ..sniff import read_zip_member, zip_names
from .text import html_to_document

CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
OPF = "{http://www.idpf.org/2007/opf}"
DC = "{http://purl.org/dc/elements/1.1/}"


class EpubConverter:
    name = "epub"
    priority = 78

    def matches(self, attachment, data, context) -> int:
        names = zip_names(data)
        return 95 if "META-INF/container.xml" in names and "mimetype" in names else 0

    def convert(self, attachment, data, context) -> Document:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            names = set(bundle.namelist())
            container = ElementTree.fromstring(
                read_zip_member(
                    bundle, "META-INF/container.xml", max_bytes=context.limits.max_bytes
                )
            )
            rootfile = container.find(f".//{CONTAINER}")
            if rootfile is None:
                raise ValueError("EPUB container has no rootfile")
            opf_path = rootfile.attrib.get("full-path", "")
            if not opf_path or opf_path not in names:
                raise ValueError("EPUB package document is missing")
            opf = ElementTree.fromstring(
                read_zip_member(bundle, opf_path, max_bytes=context.limits.max_bytes)
            )
            document = Document(source_format="epub")
            title = opf.find(f".//{DC}title")
            if title is not None and title.text:
                document.title = title.text.strip()
                document.heading(document.title, 1)
            manifest = {}
            for item in opf.findall(f".//{OPF}manifest/{OPF}item"):
                manifest[item.attrib.get("id", "")] = item.attrib.get("href", "")
            base = posixpath.dirname(opf_path)
            spine = opf.findall(f".//{OPF}spine/{OPF}itemref")
            for index, itemref in enumerate(spine, 1):
                href = manifest.get(itemref.attrib.get("idref", ""), "")
                if not href:
                    continue
                path = posixpath.normpath(posixpath.join(base, href))
                if path not in names:
                    continue
                raw = read_zip_member(bundle, path, max_bytes=context.limits.max_bytes)
                chapter = html_to_document(raw.decode("utf-8", errors="replace"))
                if chapter.blocks:
                    document.heading(f"Chapter {index}", 2)
                    document.blocks.extend(chapter.blocks)
                if len(document.blocks) > 600:
                    document.warn("Only the first part of the EPUB was converted.")
                    break
        return document
