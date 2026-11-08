"""Tests for shared channel infrastructure."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import socket
import struct
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection
from typing import ClassVar

import pytest

from sora_agent.channels import _http
from sora_agent.channels._jwt import (
    JwtError,
    _validate_claims,
    fetch_jwks,
    verify_rs256,
)
from sora_agent.channels._sse import iter_sse
from sora_agent.channels._util import Dedupe, split_ids, split_text
from sora_agent.channels._webhook import WebhookServer
from sora_agent.channels._ws import (
    WebSocketClient,
    WebSocketClosed,
    WebSocketError,
    _validate_handshake,
    encode_frame,
    mask_bytes,
    unmask,
)

# A throwaway 1024-bit RSA key generated for these tests only.
TOKEN = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InRlc3Qta2lkIn0."
    "eyJpc3MiOiJ0ZXN0LWlzc3VlciIsImF1ZCI6InRlc3QtYXVkaWVuY2UiLCJzdWIiOiJ0ZXN0"
    "ZXIiLCJleHAiOjQxMDI0NDQ4MDAsIm5iZiI6MH0."
    "stydX4tmk7PTHl-YOnLDG0KoRn06u4eLk6RMyPnQL3XtbiUzKi6pxtuBeASMmfbHPrw6xPw"
    "UEdHyv4zsMO5NjWcl2C4rMRRY2sTWrO9fDxhmxpjHJRIUpbwR5oADop-sfiWa-NNU3YEqcy"
    "4b9OdIMBFV90oeX6dnRq5KHga8cs8"
)
N = (
    "vZJlvnsNtDKz3Zu9Z9Fz3YtHFNkr3f3M1wXsQcW4j9s_heUU_Ea3hQmCEpe2NQpgjed4"
    "InHeDGLiadD1_APrCotepNjEyVyj0OnQDTRK5tuxsaWVFRPp4ou4-lIM_47uOsll_seUjAZq"
    "BVSLuUybTh77idZbdu_JHqVMAoA1v38"
)
E = "AQAB"
JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "kid": "test-kid",
            "use": "sig",
            "alg": "RS256",
            "n": N,
            "e": E,
        }
    ]
}


# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------


def test_split_ids_drops_blanks():
    assert split_ids(" 1, ,2 , ") == ("1", "2")


def test_split_text_prefers_breaks_and_hard_splits():
    assert split_text("alpha beta gamma", 10) == ["alpha beta", "gamma"]
    assert split_text("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]
    assert split_text("   ", 10) == []
    with pytest.raises(ValueError):
        split_text("x", 0)


def test_dedupe_is_bounded_and_lru():
    dedupe = Dedupe(maxsize=2)
    assert dedupe.seen("a") is False
    assert dedupe.seen("a") is True
    assert dedupe.seen("b") is False
    assert dedupe.seen("c") is False
    # "a" was the least recently used entry and has been evicted.
    assert dedupe.seen("a") is False
    assert dedupe.seen(None) is False


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


def test_iter_sse_parses_multiline_and_ignores_comments():
    events = list(
        iter_sse(
            [
                ": keepalive",
                'data: {"a": 1}',
                "",
                "data: first",
                "data: second",
                "",
                "data: not-json",
                "",
            ]
        )
    )
    assert events[0] == {"a": 1}
    assert events[1] == {"data": "first\nsecond"}
    assert events[2] == {"data": "not-json"}


# ---------------------------------------------------------------------------
# webhook server and HTTP helpers
# ---------------------------------------------------------------------------


def _start_webhook(callback, *, max_body_bytes=1024):
    server = WebhookServer(
        "127.0.0.1",
        0,
        callback,
        max_body_bytes=max_body_bytes,
    )
    server.start()
    return server


def test_webhook_server_get_post_and_404():
    def callback(method, path, headers, body):
        if path != "/ok":
            return 404, {"Content-Type": "text/plain"}, b"no"
        if method == "GET":
            return 200, {"Content-Type": "text/plain"}, b"hello"
        return 200, {"Content-Type": "application/json"}, json.dumps(
            {"body": body.decode(), "header": headers.get("x-test", "")}
        )

    server = _start_webhook(callback)
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        with urllib.request.urlopen(base + "/ok", timeout=2) as response:
            assert response.read() == b"hello"
        request = urllib.request.Request(
            base + "/ok",
            data=b"payload",
            headers={"X-Test": "yes"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload == {"body": "payload", "header": "yes"}
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/missing", timeout=2)
        assert exc.value.code == 404
        exc.value.close()
    finally:
        server.stop()


def test_webhook_server_rejects_oversized_body():
    def callback(method, path, headers, body):
        return 200, {}, b"ok"

    server = _start_webhook(callback, max_body_bytes=8)
    try:
        connection = HTTPConnection("127.0.0.1", server.bound_port, timeout=2)
        connection.request("POST", "/", body=b"x" * 64)
        response = connection.getresponse()
        assert response.status == 413
        response.read()
        connection.close()
    finally:
        server.stop()


def test_http_json_download_and_errors():
    def callback(method, path, headers, body):
        if path == "/json":
            return 200, {"Content-Type": "application/json"}, b'{"ok": true}'
        if path == "/bytes":
            return 200, {"Content-Type": "application/octet-stream"}, b"abc"
        return 404, {"Content-Type": "text/plain"}, b"missing"

    server = _start_webhook(callback)
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        assert _http.request_json(base + "/json") == {"ok": True}
        assert _http.download(base + "/bytes", max_bytes=3) == b"abc"
        with pytest.raises(_http.HttpError) as exc:
            _http.request_json(base + "/missing")
        assert exc.value.status == 404
        with pytest.raises(_http.HttpError):
            _http.download(base + "/bytes", max_bytes=2)
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# WebSocket framing and a real local round-trip
# ---------------------------------------------------------------------------


def test_websocket_masking_round_trip():
    payload = b"hello websocket"
    key = b"\x01\x02\x03\x04"
    masked = mask_bytes(payload, key)
    assert masked != payload
    assert unmask(masked, key) == payload
    frame = encode_frame(0x1, payload)
    assert frame[0] == 0x81
    assert frame[1] & 0x80


def test_websocket_handshake_validation():
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    ).decode()
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode()
    _validate_handshake(response, key)
    with pytest.raises(WebSocketError):
        _validate_handshake(response.replace(accept.encode(), b"wrong"), key)


class _WSHandler:
    received: ClassVar[list[tuple[int, bytes]]] = []

    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.handle()

    def handle(self):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.request.recv(1)
            if not chunk:
                return
            data += chunk
        key = ""
        for line in data.decode("iso-8859-1").split("\r\n"):
            name, _, value = line.partition(":")
            if name.lower() == "sec-websocket-key":
                key = value.strip()
        accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode()
        self.request.sendall(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )
        self.request.sendall(encode_frame(0x1, b'"hello"', mask=False))
        opcode, payload = _read_client_frame(self.request)
        self.received.append((opcode, payload))
        self.request.sendall(encode_frame(0x9, b"ping", mask=False))
        self.request.sendall(encode_frame(0x1, b'"bye"', mask=False))
        with contextlib.suppress(Exception):
            _read_client_frame(self.request)
        try:
            self.request.close()
        except OSError:
            pass


def _read_client_frame(sock):
    header = _recv_exact(sock, 2)
    first, second = header[0], header[1]
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if second & 0x80 else b""
    payload = _recv_exact(sock, length)
    if mask:
        payload = unmask(payload, mask)
    return opcode, payload


def _recv_exact(sock, count):
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise WebSocketClosed(1006, "closed")
        data += chunk
    return data


class _WSServer:
    def __init__(self):
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(1)
        self.port = self.socket.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        try:
            request, _address = self.socket.accept()
        except OSError:
            return
        try:
            _WSHandler(request, None, None)
        finally:
            try:
                request.close()
            except OSError:
                pass

    def stop(self):
        try:
            self.socket.close()
        except OSError:
            pass


def test_websocket_client_round_trip():
    server = _WSServer()
    client = WebSocketClient.connect(
        f"ws://127.0.0.1:{server.port}/",
        timeout=2,
    )
    try:
        assert client.recv_json(timeout=2) == "hello"
        client.send_json({"x": 1})
        assert client.recv_json(timeout=2) == "bye"
    finally:
        client.close()
        server.stop()
    assert _WSHandler.received
    opcode, payload = _WSHandler.received[0]
    assert opcode == 0x1
    assert json.loads(payload) == {"x": 1}


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def _serve_jwks():
    def callback(method, path, headers, body):
        return 200, {"Content-Type": "application/json"}, json.dumps(JWKS).encode()

    server = _start_webhook(callback)
    return server


def test_verify_rs256_accepts_a_valid_token():
    claims = verify_rs256(
        TOKEN,
        JWKS,
        audience="test-audience",
        issuer="test-issuer",
        now=1_800_000_000,
    )
    assert claims["sub"] == "tester"


def test_verify_rs256_rejects_tampering_and_wrong_claims():
    with pytest.raises(JwtError):
        verify_rs256(TOKEN[:-2] + "AA", JWKS, now=1_800_000_000)
    with pytest.raises(JwtError):
        verify_rs256(
            TOKEN,
            JWKS,
            audience="wrong",
            now=1_800_000_000,
        )
    with pytest.raises(JwtError):
        verify_rs256(
            TOKEN,
            JWKS,
            issuer="wrong",
            now=1_800_000_000,
        )


def test_validate_claims_rejects_expired_and_future_tokens():
    with pytest.raises(JwtError):
        _validate_claims(
            {"exp": 100},
            audience=None,
            issuer=None,
            now=200,
            leeway=0,
        )
    with pytest.raises(JwtError):
        _validate_claims(
            {"nbf": 300},
            audience=None,
            issuer=None,
            now=200,
            leeway=0,
        )


def test_fetch_jwks_uses_the_cache():
    calls = []

    def callback(method, path, headers, body):
        calls.append(1)
        return 200, {"Content-Type": "application/json"}, json.dumps(JWKS).encode()

    server = _start_webhook(callback)
    try:
        url = f"http://127.0.0.1:{server.bound_port}/jwks"
        first = fetch_jwks(url)
        second = fetch_jwks(url)
        assert first == second
        assert len(calls) == 1
    finally:
        server.stop()
