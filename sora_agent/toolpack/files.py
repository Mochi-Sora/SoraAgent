"""File tools: read, write, patch, and search.

These tools are registered as *hidden* entries.  ``find_tools`` is the only
route to them; the initial model tool list still contains only ``find_tools``
(and ``memory_recall`` when memory is enabled).
"""

from __future__ import annotations

import fnmatch
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

from ..tools import Tool, ToolRegistry

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
    }
)


def _read_text(path: Path, max_bytes: int) -> tuple[str, bool]:
    with path.open("rb") as handle:
        data = handle.read(max(1, int(max_bytes)) + 1)
    truncated = len(data) > max_bytes
    data = data[:max_bytes]
    if b"\x00" in data[:4096]:
        raise ValueError("binary file (NUL byte detected)")
    return data.decode("utf-8", errors="replace"), truncated


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _truncate_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + "\n...[truncated]"


def register_file_tools(registry: ToolRegistry, config: Any) -> list[Tool]:
    added: list[Tool] = []

    def read_file(
        path: Annotated[str, "File path, relative to the workspace"],
        offset: Annotated[int, "1-based first line to read"] = 1,
        limit: Annotated[int, "Maximum lines to return (1-2000)"] = 400,
    ) -> str:
        target = config.resolve(path, must_exist=True)
        if target.is_dir():
            return f"error: {target} is a directory; use search_files"
        text, truncated = _read_text(target, config.max_read_bytes)
        lines = text.splitlines()
        start = max(0, int(offset) - 1)
        end = min(len(lines), start + max(1, min(int(limit), 2000)))
        numbered = [f"{index + 1:>6}|{lines[index]}" for index in range(start, end)]
        body = "\n".join(numbered)
        if not body:
            body = "(empty or offset beyond end of file)"
        elif truncated or end < len(lines):
            body += f"\n...[truncated; next offset={end + 1}]"
        body = _truncate_chars(body, config.max_read_chars)
        return f"File: {target}\n{body}"

    def write_file(
        path: Annotated[str, "Destination file path"],
        content: Annotated[str, "Complete new file content"],
    ) -> str:
        target = config.resolve(path)
        encoded = content.encode("utf-8")
        if len(encoded) > config.max_write_bytes:
            return (
                f"error: content exceeds the {config.max_write_bytes} byte write limit"
            )
        _atomic_write(target, content)
        return f"wrote {len(encoded)} bytes to {target}"

    def patch(
        path: Annotated[str, "File path to edit"],
        old_text: Annotated[str, "Exact text to replace"],
        new_text: Annotated[str, "Replacement text"],
        replace_all: Annotated[
            bool, "Replace every occurrence instead of requiring exactly one"
        ] = False,
    ) -> str:
        if not str(old_text):
            return "error: old_text must not be empty"
        target = config.resolve(path, must_exist=True)
        text, _truncated = _read_text(target, config.max_read_bytes)
        count = text.count(old_text)
        if count == 0:
            return f"error: old_text was not found in {target}"
        if count > 1 and not replace_all:
            return (
                f"error: old_text occurs {count} times; pass replace_all=true or "
                "provide a more specific old_text"
            )
        updated = text.replace(old_text, new_text, -1 if replace_all else 1)
        if len(updated.encode("utf-8")) > config.max_write_bytes:
            return "error: edited content exceeds the write limit"
        _atomic_write(target, updated)
        removed = old_text.count("\n") + 1
        added = new_text.count("\n") + 1
        return f"patched {target} ({count if replace_all else 1} occurrence(s); -{removed} +{added} lines)"

    def search_files(
        pattern: Annotated[str, "Regex (content) or glob (files)"],
        path: Annotated[str, "File or directory to search"] = ".",
        target: Annotated[str, "Search 'content' or find 'files' by glob"] = "content",
        max_results: Annotated[int, "Maximum matches to return"] = 50,
    ) -> str:
        root = config.resolve(path, must_exist=True)
        mode = str(target or "content").strip().lower()
        if mode not in {"content", "files"}:
            return "error: target must be 'content' or 'files'"
        cap = max(1, min(int(max_results), config.max_search_results))
        results: list[str] = []
        if root.is_file():
            files = [root]
        else:
            files = []
            for directory, subdirs, filenames in os.walk(root):
                subdirs[:] = sorted(name for name in subdirs if name not in _SKIP_DIRS)
                for filename in sorted(filenames):
                    files.append(Path(directory) / filename)
        if mode == "files":
            pattern_lower = pattern.lower()
            for file_path in files:
                relative = str(
                    file_path.relative_to(root) if root.is_dir() else file_path
                )
                if fnmatch.fnmatch(relative.lower(), pattern_lower) or (
                    fnmatch.fnmatch(file_path.name.lower(), pattern_lower)
                ):
                    results.append(relative)
                    if len(results) >= cap:
                        break
            body = "\n".join(results) if results else "(no matching files)"
            return f"Files matching {pattern!r} under {root}:\n{body}"
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            return f"error: invalid regex: {exc}"
        for file_path in files:
            try:
                if file_path.stat().st_size > config.max_search_file_bytes:
                    continue
                text, _truncated = _read_text(file_path, config.max_search_file_bytes)
            except (OSError, ValueError):
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    relative = str(
                        file_path.relative_to(root) if root.is_dir() else file_path
                    )
                    results.append(f"{relative}:{line_number}: {line[:300]}")
                    if len(results) >= cap:
                        break
            if len(results) >= cap:
                break
        body = "\n".join(results) if results else "(no matches)"
        return f"Content matching {pattern!r} under {root}:\n{body}"

    definitions = [
        Tool(
            read_file,
            "read_file",
            "Read a text file with line numbers and pagination. Use instead of cat/head/tail.",
            aliases=("read file", "open file", "cat", "view file"),
        ),
        Tool(
            write_file,
            "write_file",
            "Write complete content to a file, creating parent directories. Overwrites the file.",
            needs_approval=True,
            dedupe=True,
            aliases=("write file", "create file", "save file"),
        ),
        Tool(
            patch,
            "patch",
            "Replace exact text in a file. Use instead of sed/awk for targeted edits.",
            needs_approval=True,
            dedupe=True,
            aliases=("edit file", "replace text", "find and replace"),
        ),
        Tool(
            search_files,
            "search_files",
            "Search file contents by regex or find files by glob. Use instead of grep/rg/find.",
            aliases=("search files", "grep", "find files", "ripgrep"),
        ),
    ]
    for definition in definitions:
        if not registry.has(definition.name):
            registry.add(definition)
            added.append(definition)
    return added
