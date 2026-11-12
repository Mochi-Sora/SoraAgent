"""Code-execution tool for scripted work."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Annotated, Any

from ..tools import Tool, ToolRegistry


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 80)].rstrip() + "\n...[truncated]"


def register_execution_tools(registry: ToolRegistry, config: Any) -> list[Tool]:
    added: list[Tool] = []

    def execute_code(
        code: Annotated[str, "Python source code to execute"],
        timeout: Annotated[float, "Timeout in seconds"] = 30.0,
    ) -> str:
        source = str(code or "")
        if not source.strip():
            return "error: code must not be empty"
        if len(source.encode("utf-8")) > 1_000_000:
            return "error: code exceeds the 1 MB limit"
        limit = max(0.1, min(float(timeout), config.code_timeout))
        kwargs: dict[str, Any] = {
            "cwd": str(config.root),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen([sys.executable, "-c", source], **kwargs)
        except OSError as exc:
            return f"error: could not start Python: {exc}"
        timed_out = False
        try:
            out, err = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                process.kill()
            try:
                out, err = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                out, err = "", ""
        lines = [f"exit_code: {process.poll()}"]
        if timed_out:
            lines.append("timed_out: true")
        if out:
            lines.append("stdout:\n" + _clip(out, config.max_terminal_output))
        if err:
            lines.append("stderr:\n" + _clip(err, config.max_terminal_output))
        if not out and not err:
            lines.append("(no output)")
        return "\n".join(lines)

    definition = Tool(
        execute_code,
        "execute_code",
        "Run a Python script with a timeout. Use when a task needs several tool calls or data processing.",
        needs_approval=True,
        aliases=("run python", "python script", "code", "program"),
    )
    if not registry.has(definition.name):
        registry.add(definition)
        added.append(definition)
    return added
