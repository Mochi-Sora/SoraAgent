"""Run one tool call in a child process so a hang can actually be killed.

Threads cannot be interrupted in Python, so a timeout only becomes real when the
work runs somewhere the parent can terminate.  This is crash and hang isolation,
not a security sandbox: a child process keeps the same filesystem, network, and
environment access as the gateway.

The result travels over a pipe that the parent reads while the child is alive.
This matters for large results: waiting for the child to exit first can deadlock
once the pipe buffer fills, producing a false timeout for a call that already
finished.
"""

from __future__ import annotations

import multiprocessing
import pickle
import time
from multiprocessing.connection import wait
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
        payload = value if isinstance(value, str) else str(value)
        # Fail here, where the message is meaningful, instead of in a queue
        # feeder thread after this process has already exited.
        payload = pickle.loads(pickle.dumps(payload))
        result.send(("ok", payload))
    except BaseException as exc:  # noqa: BLE001 - report failures to the parent
        try:
            result.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001 - the parent may already be gone
            pass
    finally:
        try:
            result.close()
        except OSError:
            pass


def _terminate(process: Any) -> None:
    if not process.is_alive():
        process.join(0.2)
        return
    process.terminate()
    process.join(1.0)
    if process.is_alive():
        process.kill()
        process.join(1.0)


def run_isolated(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout: float,
) -> str:
    """Call ``fn(**kwargs)`` in a spawned process and return its text result."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_entry,
        args=(fn, kwargs, sender),
        daemon=True,
    )
    try:
        process.start()
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        receiver.close()
        sender.close()
        raise ToolPicklingError(
            "tool and arguments must be picklable for process isolation"
        ) from exc
    except OSError as exc:
        receiver.close()
        sender.close()
        raise IsolationError(
            f"could not start the tool process: {exc}"
        ) from exc
    sender.close()

    try:
        ready = wait(
            [receiver, process.sentinel],
            timeout=max(0.0, float(timeout)),
        )
        if not ready:
            raise ToolTimeout(f"tool exceeded {timeout:g}s")
        if receiver in ready:
            try:
                status, payload = receiver.recv()
            except (EOFError, OSError) as exc:
                raise IsolationError(
                    f"tool process failed: {exc}"
                ) from exc
        else:
            # The child exited without writing a result (for example, a spawn
            # import error).  Drain anything it managed to send first.
            if not receiver.poll(0.1):
                raise IsolationError(
                    "tool process exited without returning a result"
                )
            try:
                status, payload = receiver.recv()
            except (EOFError, OSError) as exc:
                raise IsolationError(
                    f"tool process failed: {exc}"
                ) from exc
    finally:
        receiver.close()
        _terminate(process)

    if status == "error":
        raise IsolationError(str(payload))
    return str(payload)
