"""Shell and background-process tools."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from ..tools import Tool, ToolRegistry

log = logging.getLogger(__name__)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 80)].rstrip() + "\n...[truncated]"


@dataclass
class _Process:
    session_id: str
    process: subprocess.Popen
    command: str
    cwd: str
    started: float
    stdout: list = field(default_factory=list)
    stderr: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished_at: float = 0.0

    def _reader(self, stream: Any, target: list) -> None:
        try:
            for line in iter(stream.readline, ""):
                with self.lock:
                    target.append(line)
        finally:
            try:
                stream.close()
            except Exception:  # closing a reader is best effort
                log.debug("process stream close failed", exc_info=True)

    def start_readers(self) -> None:
        for stream, target in (
            (self.process.stdout, self.stdout),
            (self.process.stderr, self.stderr),
        ):
            if stream is None:
                continue
            thread = threading.Thread(
                target=self._reader,
                args=(stream, target),
                daemon=True,
            )
            thread.start()

    def output(self, kind: str = "both") -> str:
        with self.lock:
            out = "".join(self.stdout)
            err = "".join(self.stderr)
        if kind == "stdout":
            return out
        if kind == "stderr":
            return err
        parts = []
        if out:
            parts.append("stdout:\n" + out)
        if err:
            parts.append("stderr:\n" + err)
        return "\n".join(parts)

    def status(self) -> str:
        code = self.process.poll()
        if code is None:
            return "running"
        return f"exited({code})"

    def wait(self, timeout: float) -> int | None:
        try:
            code = self.process.wait(timeout=max(0.0, timeout))
        except subprocess.TimeoutExpired:
            return None
        self.finished_at = time.monotonic()
        return code

    def close(self) -> None:
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # closing a completed pipe is best effort
                    log.debug("process pipe close failed", exc_info=True)
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except Exception:  # closing a completed pipe is best effort
                log.debug("process stdin close failed", exc_info=True)

    def kill(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                self.process.kill()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except OSError:
            try:
                self.process.kill()
            except OSError:
                pass


class ProcessRegistry:
    def __init__(self, max_output: int = 20_000) -> None:
        self.max_output = max(200, int(max_output))
        self._items: dict[str, _Process] = {}
        self._lock = threading.Lock()

    def add(self, process: subprocess.Popen, command: str, cwd: str) -> _Process:
        session_id = "proc_" + uuid.uuid4().hex[:10]
        record = _Process(
            session_id=session_id,
            process=process,
            command=command,
            cwd=cwd,
            started=time.monotonic(),
        )
        record.start_readers()
        with self._lock:
            self._items[session_id] = record
        return record

    def get(self, session_id: str) -> _Process | None:
        with self._lock:
            return self._items.get(str(session_id or ""))

    def all(self) -> list[_Process]:
        with self._lock:
            return list(self._items.values())

    def prune(self, keep: int = 32) -> None:
        with self._lock:
            finished = [
                item for item in self._items.values() if item.process.poll() is not None
            ]
            finished.sort(key=lambda item: item.finished_at or item.started)
            for item in finished[:-keep]:
                item.close()
                self._items.pop(item.session_id, None)


def _spawn(command: str, cwd: str) -> subprocess.Popen:
    kwargs: dict[str, Any] = {
        "shell": True,
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def _collect(
    process: subprocess.Popen, timeout: float, limit: int
) -> tuple[int | None, str, str, bool]:
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
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
    code = process.poll()
    return code, _clip(out or "", limit), _clip(err or "", limit), timed_out


def register_shell_tools(registry: ToolRegistry, config: Any) -> list[Tool]:
    processes = ProcessRegistry(max_output=config.max_terminal_output)
    added: list[Tool] = []

    def terminal(
        command: Annotated[str, "Shell command to execute"],
        cwd: Annotated[str, "Working directory"] = ".",
        timeout: Annotated[float, "Timeout in seconds"] = 120.0,
        background: Annotated[
            bool, "Start in the background and return a session id"
        ] = False,
    ) -> str:
        directory = str(config.resolve(cwd, must_exist=True))
        if not os.path.isdir(directory):
            return f"error: cwd is not a directory: {directory}"
        effective_timeout = max(0.1, min(float(timeout), config.terminal_timeout))
        try:
            process = _spawn(str(command), directory)
        except OSError as exc:
            return f"error: could not start command: {exc}"
        if background:
            record = processes.add(process, str(command), directory)
            processes.prune()
            return (
                "started background process\n"
                f"session_id: {record.session_id}\npid: {process.pid}\ncommand: {command}"
            )
        code, out, err, timed_out = _collect(
            process, effective_timeout, config.max_terminal_output
        )
        lines = [
            f"exit_code: {code}",
            f"cwd: {directory}",
        ]
        if timed_out:
            lines.append("timed_out: true")
        if out:
            lines.append("stdout:\n" + out)
        if err:
            lines.append("stderr:\n" + err)
        if not out and not err:
            lines.append("(no output)")
        return "\n".join(lines)

    def process_manage(
        action: Annotated[
            Literal["list", "poll", "log", "wait", "kill", "write"],
            "Background-process action",
        ],
        session_id: Annotated[str, "Process session id"] = "",
        data: Annotated[str, "Data to write to stdin"] = "",
        timeout: Annotated[float, "Wait timeout in seconds"] = 30.0,
    ) -> str:
        mode = str(action).strip().lower()
        if mode == "list":
            items = sorted(processes.all(), key=lambda item: item.started, reverse=True)
            if not items:
                return "no background processes"
            lines = []
            for item in items:
                lines.append(
                    f"{item.session_id} | {item.status()} | pid={item.process.pid} | {item.command[:120]} | {item.cwd}"
                )
            return "\n".join(lines)
        record = processes.get(session_id)
        if record is None:
            return f"error: unknown session_id: {session_id}"
        if mode == "poll":
            return f"{record.session_id} | {record.status()}\n{_clip(record.output(), config.max_terminal_output)}"
        if mode == "log":
            return _clip(record.output(), config.max_terminal_output)
        if mode == "wait":
            code = record.wait(min(float(timeout), 600.0))
            if code is None:
                return f"still running: {record.session_id}"
            record.close()
            return f"exit_code: {code}\n{_clip(record.output(), config.max_terminal_output)}"
        if mode == "kill":
            record.kill()
            record.close()
            return f"killed {record.session_id}"
        if mode == "write":
            if record.process.stdin is None:
                return "error: process stdin is not available"
            try:
                record.process.stdin.write(str(data))
                record.process.stdin.flush()
            except (OSError, ValueError) as exc:
                return f"error: could not write to stdin: {exc}"
            return f"wrote {len(str(data))} bytes to {record.session_id}"
        return f"error: unknown action: {action}"

    definitions = [
        Tool(
            terminal,
            "terminal",
            "Execute a shell command with a timeout and capped output. Use read_file/search_files instead of cat/grep.",
            needs_approval=True,
            aliases=("shell", "run command", "bash", "exec"),
        ),
        Tool(
            process_manage,
            "process_manage",
            "Manage background processes started with terminal(background=true): list, poll, log, wait, kill, write.",
            needs_approval=True,
            aliases=("process", "background process", "kill process"),
        ),
    ]
    for definition in definitions:
        if not registry.has(definition.name):
            registry.add(definition)
            added.append(definition)
    return added
