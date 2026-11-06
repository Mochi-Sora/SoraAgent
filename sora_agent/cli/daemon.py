"""Background lifecycle management for the chat gateway.

``sora gateway start`` launches ``python -m sora_agent.channels`` in a new
session, writes a PID file, and redirects output to a log.  The detached child
is not attached to the SSH terminal, so it survives the connection closing.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..settings import DEFAULT_ROOT
from . import ui

PID_ENV = "SORA_GATEWAY_PID"
LOG_ENV = "SORA_GATEWAY_LOG"
DEFAULT_PID = Path.home() / ".sora_agent" / "gateway.pid"
DEFAULT_LOG = Path.home() / ".sora_agent" / "gateway.log"
STOP_TIMEOUT = 15.0
START_GRACE = 1.5


@dataclass(frozen=True)
class GatewayProcess:
    pid: int
    started_at: str
    version: str


def pid_path() -> Path:
    value = os.getenv(PID_ENV, "").strip()
    return Path(value).expanduser() if value else DEFAULT_PID


def log_path() -> Path:
    value = os.getenv(LOG_ENV, "").strip()
    return Path(value).expanduser() if value else DEFAULT_LOG


def _read_process() -> GatewayProcess | None:
    try:
        raw = pid_path().read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("pid")
    if not isinstance(value, (int, str)):
        return None
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    return GatewayProcess(
        pid=pid,
        started_at=str(data.get("started_at") or ""),
        version=str(data.get("version") or ""),
    )


def _write_process(pid: int) -> None:
    path = pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _remove_process() -> None:
    try:
        pid_path().unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        ui.warning(f"Could not remove {pid_path()}: {exc}")


def _windows_pid_alive(pid: int) -> bool:
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = windll.kernel32
    handle = kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _looks_like_gateway(pid: int) -> bool:
    if os.name == "nt":
        return True
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    command = result.stdout.strip()
    return "sora_agent.channels" in command or "sora-gateway" in command


def _wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


def _force_signal() -> int:
    return getattr(signal, "SIGKILL", signal.SIGTERM)


def _tail(path: Path, lines: int) -> list[str]:
    count = max(1, int(lines))
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= count:
            step = min(block, size)
            size -= step
            handle.seek(size)
            data = handle.read(step) + data
    return data.decode("utf-8", "replace").splitlines()[-count:]


def _print_log_tail(lines: int = 20) -> None:
    path = log_path()
    if not path.is_file():
        ui.info(f"No log file at {path}.")
        return
    try:
        tail = _tail(path, lines)
    except OSError as exc:
        ui.warning(f"Could not read {path}: {exc}")
        return
    ui.info(f"Last {len(tail)} log line(s) from {path}:")
    for line in tail:
        print(line)


def start_gateway() -> int:
    record = _read_process()
    if record is not None and _pid_alive(record.pid):
        if _looks_like_gateway(record.pid):
            ui.warning(f"Gateway is already running (pid {record.pid}).")
            ui.info(f"Logs: {log_path()}")
            return 0
        ui.error(
            f"{pid_path()} refers to pid {record.pid}, which does not look "
            "like a Sora gateway. Remove the file if it is stale."
        )
        return 2
    if record is not None or pid_path().exists():
        ui.info("Removing a stale gateway pid file.")
        _remove_process()

    log = log_path()
    try:
        parent_existed = log.parent.exists()
        log.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed or log.parent == DEFAULT_ROOT:
            try:
                os.chmod(log.parent, 0o700)
            except OSError:
                pass
        output = log.open("ab")
        try:
            os.chmod(log, 0o600)
        except OSError:
            pass
    except OSError as exc:
        ui.error(f"Could not open {log}: {exc}")
        return 2

    command = [sys.executable, "-m", "sora_agent.channels"]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        ) | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        options["start_new_session"] = True

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with output:
            output.write(
                f"\n=== {started} sora gateway start ===\n".encode()
            )
            output.flush()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                close_fds=True,
                **options,
            )
    except OSError as exc:
        ui.error(f"Could not start the gateway: {exc}")
        return 2

    try:
        exit_code = process.wait(timeout=START_GRACE)
    except subprocess.TimeoutExpired:
        pass
    else:
        ui.error(f"Gateway exited during startup (code {exit_code}).")
        _print_log_tail()
        return 1

    try:
        _write_process(process.pid)
    except OSError as exc:
        ui.error(
            f"Gateway started as pid {process.pid}, but the pid file could "
            f"not be written: {exc}"
        )
        return 2

    ui.success(f"Gateway started in the background (pid {process.pid}).")
    ui.info(f"Logs: {log}")
    ui.info("Stop with `sora gateway stop`.")
    return 0


def stop_gateway() -> int:
    record = _read_process()
    if record is None:
        if pid_path().exists():
            ui.warning(f"Ignoring an invalid pid file at {pid_path()}.")
            _remove_process()
        else:
            ui.info("Gateway is not running.")
        return 0

    if not _pid_alive(record.pid):
        _remove_process()
        ui.info("Gateway is not running (removed a stale pid file).")
        return 0

    if not _looks_like_gateway(record.pid):
        ui.error(
            f"pid {record.pid} does not look like a Sora gateway; refusing "
            f"to stop it. Remove {pid_path()} if it is stale."
        )
        return 2

    ui.info(f"Stopping gateway (pid {record.pid})...")
    try:
        os.kill(record.pid, signal.SIGTERM)
    except OSError as exc:
        ui.error(f"Could not signal pid {record.pid}: {exc}")
        return 2

    if _wait_for_exit(record.pid, STOP_TIMEOUT):
        _remove_process()
        ui.success("Gateway stopped.")
        return 0

    ui.warning("Gateway did not stop in time; forcing it.")
    try:
        os.kill(record.pid, _force_signal())
    except OSError as exc:
        ui.error(f"Could not force-stop pid {record.pid}: {exc}")
        return 2
    if not _wait_for_exit(record.pid, 3.0):
        ui.error(f"Gateway is still running (pid {record.pid}).")
        return 2
    _remove_process()
    ui.success("Gateway stopped.")
    return 0


def status_gateway() -> int:
    record = _read_process()
    if record is None:
        if pid_path().exists():
            ui.warning(f"Invalid gateway pid file at {pid_path()}.")
            return 2
        ui.info("Gateway is not running.")
        ui.info("Start it with `sora gateway start`.")
        return 1

    if not _pid_alive(record.pid):
        ui.warning(f"Gateway is not running (stale pid {record.pid}).")
        return 1

    if not _looks_like_gateway(record.pid):
        ui.warning(
            f"pid {record.pid} is alive but does not look like a Sora gateway."
        )
        return 2

    version = f", v{record.version}" if record.version else ""
    started = f", started {record.started_at}" if record.started_at else ""
    ui.success(f"Gateway is running (pid {record.pid}{version}{started}).")
    ui.info(f"Logs: {log_path()}")
    return 0


def restart_gateway() -> int:
    code = stop_gateway()
    if code != 0:
        return code
    return start_gateway()


def show_logs(lines: int = 50) -> int:
    path = log_path()
    if not path.is_file():
        ui.info(f"No log file at {path} yet.")
        return 1
    try:
        tail = _tail(path, lines)
    except OSError as exc:
        ui.error(f"Could not read {path}: {exc}")
        return 2
    if not tail:
        ui.info(f"{path} is empty.")
        return 0
    for line in tail:
        print(line)
    return 0
