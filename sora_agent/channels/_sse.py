"""Server-sent-events parsing used by the Signal adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


def iter_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per SSE event.

    ``signal-cli`` sends JSON objects on ``data:`` lines.  Non-JSON payloads are
    yielded as ``{"data": <text>}`` so a malformed event never hides later ones.
    """
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                try:
                    value = json.loads(payload)
                except ValueError:
                    value = {"data": payload}
                if isinstance(value, dict):
                    yield value
                else:
                    yield {"data": value}
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data.append(value)
        # ``event``, ``id``, and ``retry`` fields do not change the payload.
    if data:
        payload = "\n".join(data)
        try:
            value = json.loads(payload)
        except ValueError:
            value = {"data": payload}
        if isinstance(value, dict):
            yield value
        else:
            yield {"data": value}
