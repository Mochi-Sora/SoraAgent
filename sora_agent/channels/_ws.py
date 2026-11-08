"""A small RFC 6455 WebSocket client (text frames only).

Discord's gateway and Slack Socket Mode both speak WebSocket.  Keeping this
client here avoids a dependency on a WebSocket package while still giving the
channel adapters a tested, bounded implementation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import threading
import urllib.parse
from collections.abc import Mapping
from typing import Any

from ..net import ssl_context

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_HANDSHAKE_BYTES = 64 * 1024
_OP_CONT = 0x0
_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


class WebSocketError(RuntimeError):
    """The WebSocket handshake or frame stream was invalid."""


class WebSocketClosed(WebSocketError):
    """The peer closed the WebSocket."""

    def __init__(self, code: int, reason: str = "") -> None:
        super().__init__(f"websocket closed ({code}) {reason}".strip())
        self.code = code
        self.reason = reason


class WebSocketClient:
    """A blocking WebSocket client for one text/binary frame stream."""

    def __init__(
        self,
        sock: socket.socket,
        *,
        max_frame_bytes: int = 8 * 1024 * 1024,
        log=None,
    ) -> None:
        self._sock = sock
        self._max_frame_bytes = max(1024, int(max_frame_bytes))
        self._send_lock = threading.Lock()
        self._closed = False
        self._fragment_opcode = _OP_TEXT
        self._fragments: list[bytes] = []
        self.log = log

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        max_frame_bytes: int = 8 * 1024 * 1024,
        log=None,
    ) -> WebSocketClient:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in {"ws", "wss"}:
            raise WebSocketError(f"unsupported websocket scheme: {parsed.scheme}")
        host = parsed.hostname
        if not host:
            raise WebSocketError(f"websocket url has no host: {url}")
        port = parsed.port or (443 if scheme == "wss" else 80)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            if scheme == "wss":
                sock = ssl_context().wrap_socket(sock, server_hostname=host)
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            request_headers = {
                "Host": host if port in (80, 443) else f"{host}:{port}",
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Key": key,
                "Sec-WebSocket-Version": "13",
            }
            for name, value in (headers or {}).items():
                request_headers[str(name)] = str(value)
            lines = [f"GET {target} HTTP/1.1"]
            lines.extend(f"{name}: {value}" for name, value in request_headers.items())
            lines.extend(["", ""])
            sock.sendall("\r\n".join(lines).encode("ascii"))
            response = _read_handshake(sock)
            _validate_handshake(response, key)
            sock.settimeout(None)
            return cls(
                sock,
                max_frame_bytes=max_frame_bytes,
                log=log,
            )
        except Exception:
            try:
                sock.close()
            except OSError:
                pass
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def send_text(self, text: str) -> None:
        self._send_frame(_OP_TEXT, text.encode("utf-8"))

    def send_json(self, payload: Any) -> None:
        self.send_text(json.dumps(payload, separators=(",", ":")))

    def ping(self, payload: bytes = b"") -> None:
        self._send_frame(_OP_PING, payload)

    def pong(self, payload: bytes = b"") -> None:
        self._send_frame(_OP_PONG, payload)

    def recv(self) -> tuple[int, bytes]:
        """Read one complete data frame, handling ping/pong and fragments."""
        while True:
            opcode, fin, payload = self._read_frame()
            if opcode == _OP_CLOSE:
                code, reason = _parse_close(payload)
                with self._send_lock:
                    if not self._closed:
                        self._closed = True
                        try:
                            self._write_frame(_OP_CLOSE, payload[:125])
                        except OSError:
                            pass
                try:
                    self._sock.close()
                except OSError:
                    pass
                raise WebSocketClosed(code, reason)
            if opcode == _OP_PING:
                self.pong(payload)
                continue
            if opcode == _OP_PONG:
                continue
            if opcode in (_OP_TEXT, _OP_BINARY):
                if self._fragments:
                    raise WebSocketError("new data frame inside a fragmented message")
                if fin:
                    return opcode, payload
                self._fragment_opcode = opcode
                self._fragments = [payload]
                continue
            if opcode == _OP_CONT:
                if not self._fragments:
                    raise WebSocketError("continuation frame without a message")
                self._fragments.append(payload)
                total = sum(len(part) for part in self._fragments)
                if total > self._max_frame_bytes:
                    raise WebSocketError("fragmented websocket message is too large")
                if fin:
                    data = b"".join(self._fragments)
                    self._fragments = []
                    return self._fragment_opcode, data
                continue
            raise WebSocketError(f"unsupported websocket opcode: {opcode}")

    def recv_json(self, *, timeout: float | None = None) -> Any:
        self._sock.settimeout(timeout)
        opcode, payload = self.recv()
        if opcode != _OP_TEXT:
            raise WebSocketError("expected a text websocket frame")
        try:
            return json.loads(payload.decode("utf-8"))
        except ValueError as exc:
            raise WebSocketError("websocket peer sent invalid JSON") from exc

    def close(self, code: int = 1000, reason: str = "") -> None:
        payload = struct.pack("!H", int(code)) + reason.encode("utf-8")[:123]
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._write_frame(_OP_CLOSE, payload)
            except OSError:
                pass
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        with self._send_lock:
            if self._closed:
                raise WebSocketClosed(1000, "client is closed")
            self._write_frame(opcode, payload)

    def _write_frame(self, opcode: int, payload: bytes) -> None:
        self._sock.sendall(encode_frame(opcode, payload))

    def _read_frame(self) -> tuple[int, bool, bytes]:
        header = self._read_exact(2)
        first, second = header[0], header[1]
        fin = bool(first & 0x80)
        rsv = first & 0x70
        opcode = first & 0x0F
        if rsv:
            raise WebSocketError("unsupported websocket extension bits")
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        if length > self._max_frame_bytes:
            raise WebSocketError(
                f"websocket frame is {length} bytes, above the "
                f"{self._max_frame_bytes} byte limit"
            )
        if opcode >= 0x8 and (not fin or length > 125):
            raise WebSocketError("invalid websocket control frame")
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length)
        if masked and mask:
            payload = unmask(payload, mask)
        return opcode, fin, payload

    def _read_exact(self, count: int) -> bytes:
        if count <= 0:
            return b""
        parts: list[bytes] = []
        remaining = count
        while remaining:
            try:
                chunk = self._sock.recv(remaining)
            except socket.timeout:
                raise
            except OSError as exc:
                raise WebSocketClosed(1006, str(exc)) from exc
            if not chunk:
                raise WebSocketClosed(1006, "connection closed while reading")
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)


def encode_frame(opcode: int, payload: bytes, *, mask: bool = True) -> bytes:
    """Encode one client-to-server frame."""
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(payload)
    if length < 126:
        header.append((0x80 if mask else 0) | length)
    elif length < 65536:
        header.append((0x80 if mask else 0) | 126)
        header += struct.pack("!H", length)
    else:
        header.append((0x80 if mask else 0) | 127)
        header += struct.pack("!Q", length)
    if mask:
        key = os.urandom(4)
        header += key
        return bytes(header) + mask_bytes(payload, key)
    return bytes(header) + payload


def mask_bytes(payload: bytes, key: bytes) -> bytes:
    if len(key) != 4:
        raise ValueError("websocket mask key must be four bytes")
    return bytes(byte ^ key[index % 4] for index, byte in enumerate(payload))


def unmask(payload: bytes, key: bytes) -> bytes:
    return mask_bytes(payload, key)


def _read_handshake(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1)
        if not chunk:
            raise WebSocketError("connection closed during websocket handshake")
        data += chunk
        if len(data) > _MAX_HANDSHAKE_BYTES:
            raise WebSocketError("websocket handshake response is too large")
    return bytes(data)


def _validate_handshake(response: bytes, key: str) -> None:
    head, _, _body = response.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    if not lines or " 101 " not in f" {lines[0]} ":
        raise WebSocketError(f"websocket upgrade failed: {lines[0] if lines else ''}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    if headers.get("upgrade", "").lower() != "websocket":
        raise WebSocketError("websocket upgrade response is missing Upgrade: websocket")
    if "upgrade" not in headers.get("connection", "").lower():
        raise WebSocketError("websocket upgrade response is missing Connection: Upgrade")
    expected = base64.b64encode(
        hashlib.sha1((key + _GUID).encode("ascii")).digest()
    ).decode("ascii")
    if headers.get("sec-websocket-accept") != expected:
        raise WebSocketError("websocket accept key does not match")


def _parse_close(payload: bytes) -> tuple[int, str]:
    if len(payload) < 2:
        return 1005, ""
    code = struct.unpack("!H", payload[:2])[0]
    reason = payload[2:].decode("utf-8", "replace")
    return code, reason


def websocket_ssl_context() -> ssl.SSLContext:
    """Expose the shared CA context for tests and callers."""
    return ssl_context()
