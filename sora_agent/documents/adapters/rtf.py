"""Small, dependency-free RTF text extractor.

This intentionally handles the common text subset (unicode escapes, hex
escapes, paragraph/line controls) and skips destination groups such as font
and colour tables. It is not a full RTF layout engine.
"""

from __future__ import annotations

import re

from ..model import Document
from ..sniff import is_rtf, suffix

_IGNORE_DESTINATIONS = {
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object",
    "header", "footer", "headerl", "headerr", "footerl", "footerr",
    "fldinst", "themedata", "colorschememapping", "latentstyles",
}


class RtfConverter:
    name = "rtf"
    priority = 65

    def matches(self, attachment, data, context) -> int:
        if suffix(attachment) == ".rtf" or is_rtf(data):
            return 80
        return 0

    def convert(self, attachment, data, context) -> Document:
        text = _rtf_to_text(data.decode("latin-1", errors="replace"))
        document = Document(source_format="rtf")
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if paragraph:
                document.paragraph(paragraph)
        if not document.blocks:
            document.note("The RTF contained no extractable text.")
        return document


def _rtf_to_text(source: str) -> str:
    out: list[str] = []
    stack: list[dict[str, bool]] = [{"skip": False}]
    i = 0
    length = len(source)
    while i < length:
        char = source[i]
        if char == "{":
            stack.append(dict(stack[-1]))
            i += 1
            continue
        if char == "}":
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue
        if char != "\\":
            if not stack[-1]["skip"] and char not in "\r\n":
                out.append(char)
            i += 1
            continue
        i += 1
        if i >= length:
            break
        marker = source[i]
        if marker in "\\{}":
            if not stack[-1]["skip"]:
                out.append(marker)
            i += 1
            continue
        if marker in "\r\n":
            # Some writers (notably Apple's textutil) use a backslash-newline
            # pair as a hard line break.
            if not stack[-1]["skip"]:
                out.append("\n")
            i += 1
            if marker == "\r" and i < length and source[i] == "\n":
                i += 1
            continue
        if marker == "'":
            hex_value = source[i + 1 : i + 3]
            try:
                if not stack[-1]["skip"]:
                    out.append(bytes.fromhex(hex_value).decode("cp1252", "replace"))
            except ValueError:
                pass
            i += 3
            continue
        if marker == "*":
            stack[-1]["skip"] = True
            i += 1
            continue
        match = re.match(r"([a-zA-Z]+)(-?\d+)? ?", source[i:])
        if not match:
            i += 1
            continue
        word, argument = match.group(1), match.group(2)
        i += match.end()
        if word in _IGNORE_DESTINATIONS:
            stack[-1]["skip"] = True
            continue
        if word in {"par", "line", "sect", "page"}:
            if not stack[-1]["skip"]:
                out.append("\n\n" if word in {"par", "sect", "page"} else "\n")
        elif word == "tab":
            if not stack[-1]["skip"]:
                out.append("\t")
        elif word == "u" and argument is not None:
            try:
                code = int(argument)
                if code < 0:
                    code += 65536
                if not stack[-1]["skip"]:
                    out.append(chr(code))
            except ValueError:
                pass
    text = "".join(out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
