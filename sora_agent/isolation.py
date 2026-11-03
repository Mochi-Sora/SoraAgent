"""Run one tool call in a child process so a hang can actually be killed.

Threads cannot be interrupted in Python, so a timeout only becomes real when the
work runs somewhere the parent can terminate.  This is crash and hang isolation,
not a security sandbox: a child process keeps the same filesystem, network, and
environment access as the gateway.
"""

from __future__ import annotations

import multiprocessing
import pickle
import queue
import time
from typing import Any, Callable


class IsolationError(RuntimeError):
    """The isolated call failed before producing a result."""


class ToolTimeout(IsolationError):
    """The isolated call exceeded its deadline and was terminated."""


class ToolPicklingError(IsolationError):
    """The tool or its arguments cannot cross a process boundary."""


def block_forever() -> None:
    """A picklable no-argument function used to exercise timeouts."""
    while True:
        time.sleep(0.1)


def _child_entry(
    fn: Callable[..., Any], kwargs: dict[str, Any], result: Any
) -> None:
    try:
        value = fn(**kwargs)
    except BaseException as exc:  # noqa: BLE001 - report failures to the parent
        result.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    result.put(("ok", value if isinstance(value, str) else str(value)))


def run_isolated(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout: float,
) -> str:
    """Call ``fn(**kwargs)`` in a spawned process and return its text result."""
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(
        target=_child_entry,
        args=(fn, kwargs, result),
        daemon=True,
    )
    try:
        process.start()
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ToolPicklingError(
            "tool and arguments must be picklable for process isolation"
        ) from exc
    process.join(max(0.0, float(timeout)))
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
        result.close()
        raise ToolTimeout(f"tool exceeded {timeout:g}s")
    try:
        status, payload = result.get(timeout=1.0)
    except queue.Empty as exc:
        raise IsolationError(
            "tool process exited without returning a result"
        ) from exc
    except (EOFError, OSError) as exc:
        raise IsolationError(f"tool process failed: {exc}") from exc
    finally:
        result.close()
    if status == "error":
        raise IsolationError(str(payload))
    return str(payload)
