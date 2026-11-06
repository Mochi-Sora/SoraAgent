"""Telegram Bot API transport built on the standard library.

Keeping the HTTP layer here means the channel has no required dependency and
tests can substitute a fake transport without patching ``urllib``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from ...net import urlopen


class TelegramError(RuntimeError):
    """A Telegram request failed or returned ``ok: false``."""


class TelegramTransport(Protocol):
    def get_updates(self, offset: int, timeout: int) -> list[dict[str, Any]]:
        """Long-poll for updates starting at ``offset``."""
        ...

    def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> str | None:
        """Send one message and return its platform id when available."""

    def edit_message_text(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Replace the text of a message already sent."""

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        """Acknowledge a button press so Telegram stops the spinner."""

    def clear_reply_markup(self, chat_id: str, message_id: str) -> None:
        """Remove the inline keyboard from a message."""

    def download_file(self, file_id: str, max_bytes: int) -> bytes:
        """Download one attachment, raising when it exceeds ``max_bytes``."""
        ...


class UrllibTelegramTransport:
    """Minimal Bot API client: JSON in, JSON out, no bot framework."""

    def __init__(
        self,
        token: str,
        api_base: str = "https://api.telegram.org",
        *,
        request_timeout: float = 40.0,
    ) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._request_timeout = float(request_timeout)

    def get_updates(self, offset: int, timeout: int) -> list[dict[str, Any]]:
        payload = {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
            "limit": 100,
        }
        # The HTTP timeout must outlast the server-side long poll.
        result = self._call(
            "getUpdates",
            payload,
            timeout=self._request_timeout + max(0, int(timeout)),
        )
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

    def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> str | None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if reply_to:
            try:
                payload["reply_parameters"] = {"message_id": int(reply_to)}
            except (TypeError, ValueError):
                pass
        result = self._call("sendMessage", payload)
        if isinstance(result, dict) and result.get("message_id") is not None:
            return str(result["message_id"])
        return None

    def edit_message_text(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        try:
            numeric_id = int(message_id)
        except (TypeError, ValueError):
            return
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": numeric_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        self._call("editMessageText", payload)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        self._call("answerCallbackQuery", payload)

    def clear_reply_markup(self, chat_id: str, message_id: str) -> None:
        try:
            numeric_id = int(message_id)
        except (TypeError, ValueError):
            return
        self._call(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": numeric_id,
                "reply_markup": {"inline_keyboard": []},
            },
        )

    def download_file(self, file_id: str, max_bytes: int) -> bytes:
        result = self._call("getFile", {"file_id": file_id})
        if not isinstance(result, dict) or not result.get("file_path"):
            raise TelegramError("telegram getFile returned no file path")
        reported = _as_int(result.get("file_size"))
        if reported is not None and reported > max_bytes:
            raise TelegramError(
                f"file is {reported} bytes, above the {max_bytes} byte limit"
            )
        path = urllib.parse.quote(str(result["file_path"]), safe="/")
        url = f"{self._api_base}/file/bot{self._token}/{path}"
        request = urllib.request.Request(url)
        chunks: list[bytes] = []
        total = 0
        try:
            with urlopen(request, self._request_timeout) as response:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise TelegramError(
                            f"file is above the {max_bytes} byte limit"
                        )
                    chunks.append(chunk)
        except TelegramError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise TelegramError(
                f"telegram file download failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise TelegramError(
                f"telegram file download failed: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise TelegramError("telegram file download timed out") from exc
        return b"".join(chunks)

    def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self._api_base}/bot{self._token}/{method}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout or self._request_timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise TelegramError(
                f"telegram {method} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise TelegramError(
                f"telegram {method} failed: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise TelegramError(f"telegram {method} timed out") from exc
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise TelegramError(
                f"telegram {method} returned invalid JSON"
            ) from exc
        if not isinstance(body, dict) or not body.get("ok"):
            description = body.get("description") if isinstance(body, dict) else None
            raise TelegramError(
                f"telegram {method} failed: {description or 'unknown error'}"
            )
        return body.get("result")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
