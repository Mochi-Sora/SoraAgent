"""A tiny blocking HTTP webhook server used by webhook-based channels.

The server runs one daemon thread and hands each request to a callback.  It is
deliberately small: channels only need a bounded JSON POST endpoint, not a web
framework.
"""

from __future__ import annotations

import logging
import threading
import urllib.parse
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Union

WebhookReply = tuple[int, Mapping[str, str], Union[bytes, str]]
WebhookHandler = Callable[[str, str, Mapping[str, str], bytes], WebhookReply]


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler_class(callback: WebhookHandler, max_body_bytes: int, log):
    class Handler(BaseHTTPRequestHandler):
        timeout = 30

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            try:
                length_raw = self.headers.get("Content-Length", "0")
                try:
                    length = int(length_raw)
                except (TypeError, ValueError):
                    length = -1
                if length < 0:
                    self._reply(400, {"Content-Type": "text/plain"}, b"bad length")
                    return
                if length > max_body_bytes:
                    self._reply(
                        413,
                        {"Content-Type": "text/plain"},
                        b"request body too large",
                    )
                    return
                body = self.rfile.read(length) if length else b""
                headers = {
                    str(key).lower(): str(value)
                    for key, value in self.headers.items()
                }
                status, response_headers, response_body = callback(
                    method, self.path, headers, body
                )
                if isinstance(response_body, str):
                    response_body = response_body.encode("utf-8")
                self._reply(status, response_headers, response_body)
            except Exception:
                log.exception("webhook request failed")
                self._reply(
                    500,
                    {"Content-Type": "text/plain"},
                    b"internal webhook error",
                )

        def _reply(
            self,
            status: int,
            headers: Mapping[str, str],
            body: bytes,
        ) -> None:
            self.send_response(status)
            sent = {str(key).lower() for key in headers}
            for key, value in headers.items():
                self.send_header(str(key), str(value))
            if "content-type" not in sent:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            log.debug("webhook: " + format, *args)

    return Handler


class WebhookServer:
    """Run a callback behind ``host:port`` on a background thread."""

    def __init__(
        self,
        host: str,
        port: int,
        callback: WebhookHandler,
        *,
        max_body_bytes: int = 2 * 1024 * 1024,
        logger: logging.Logger | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.callback = callback
        self.max_body_bytes = max(1, int(max_body_bytes))
        self.log = logger or logging.getLogger(__name__)
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def bound_port(self) -> int:
        if self._server is None:
            return self.port
        return int(self._server.server_address[1])

    def start(self) -> None:
        if self._server is not None:
            return
        handler = _handler_class(self.callback, self.max_body_bytes, self.log)
        self._server = _Server((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"webhook:{self.host}:{self.bound_port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)


def path_only(raw_path: str) -> str:
    """Return the path component of a request target."""
    return urllib.parse.urlsplit(raw_path).path


def query_params(raw_path: str) -> dict[str, str]:
    """Parse query parameters from a request target."""
    parsed = urllib.parse.urlsplit(raw_path)
    return {
        key: value
        for key, values in urllib.parse.parse_qs(parsed.query).items()
        for value in values[:1]
    }
