"""Optional OpenAI-compatible wake-gate model call."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .config import MemoryConfig
from .net import ssl_context


class WakeUnavailable(RuntimeError):
    pass


class OpenAIWakeGate:
    def __init__(self, config: MemoryConfig) -> None:
        if not config.wake_model or not config.wake_base_url:
            raise WakeUnavailable("wake model is not configured")
        self.config = config

    def __call__(self, report: dict[str, Any]) -> str:
        prompt = _prompt(report)
        payload = {
            "model": self.config.wake_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You review a memory-consolidation report. Propose only "
                        "bounded, reversible changes. Do not invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        url = self.config.wake_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.wake_api_key:
            headers["Authorization"] = "Bearer " + self.config.wake_api_key
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.config.wake_timeout, context=ssl_context()
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise WakeUnavailable("wake model returned no choices")
        message = choices[0].get("message", {})
        return str(message.get("content") or "")


def _prompt(report: dict[str, Any]) -> str:
    lines = [
        "Dream report:",
        "run_id: {}".format(report.get("run_id", "")),
        "transferred_days: {}".format(report.get("transferred_days", [])),
        "conflicts: {}".format(report.get("conflicts", [])),
        "communities: {}".format(report.get("communities", [])),
        "inferences: {}".format(report.get("inferences", [])),
        "errors: {}".format(report.get("errors", [])),
        "actions: {}".format(report.get("actions", [])),
        "",
        "Return a short review. If no action is needed, say 'no action'.",
    ]
    return "\n".join(lines)
