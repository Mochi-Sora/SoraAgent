"""Download, start, and stop the loopback CortexDB sidecar."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import secrets
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .client import CortexDBClient
from .config import MemoryConfig
from .net import ssl_context


class SidecarError(RuntimeError):
    pass


class SidecarManager:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None
        self._client: CortexDBClient | None = None
        self._http_addr = ""
        self._grpc_addr = ""
        self._token = ""

    @property
    def client(self) -> CortexDBClient | None:
        return self._client

    def ensure_running(self) -> CortexDBClient:
        if self._client is not None and self._client.health():
            return self._client
        self.config.ensure_directories()
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._client = None
        if self._client is not None:
            if self._client.health():
                return self._client
            self._client = None
            if self._process is not None and self._process.poll() is None:
                self.stop()
        binary = self._ensure_binary()
        self._token = self.config.token or self._load_or_create_token()
        self._http_addr = self._resolve_addr(self.config.http_addr)
        self._grpc_addr = self._resolve_addr(self.config.grpc_addr)
        env = self._child_env()
        log_path = self.config.sidecar_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab", buffering=0) as log:
            kwargs: dict[str, Any] = {
                "cwd": str(self.config.memory_dir),
                "env": env,
                "stdout": log,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            else:
                kwargs["start_new_session"] = True
            self._process = subprocess.Popen([str(binary)], **kwargs)
        client = CortexDBClient(
            self._http_addr, self._token, timeout=self.config.request_timeout
        )
        self._client = client
        deadline = time.monotonic() + max(1.0, self.config.sidecar_start_timeout)
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                message = self._log_tail()
                self._client = None
                raise SidecarError(
                    "cortexdb-grpc exited during startup"
                    + (": " + message if message else "")
                )
            if client.health():
                return client
            time.sleep(0.1)
        self.stop()
        raise SidecarError(
            "cortexdb-grpc did not become healthy at " + self._http_addr
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._client = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        except OSError:
            pass

    def health(self) -> bool:
        if self._client is None:
            return False
        return self._client.health()

    def doctor(self) -> dict[str, Any]:
        binary = self.config.sidecar_path
        return {
            "enabled": self.config.enabled,
            "binary": str(binary),
            "binary_present": binary.exists(),
            "running": self._process is not None and self._process.poll() is None,
            "healthy": self.health(),
            "http_addr": self._http_addr,
            "grpc_addr": self._grpc_addr,
            "db_path": str(self.config.db_path),
            "backup_dir": str(self.config.backup_dir),
            "version": self.config.sidecar_version,
            "log": str(self.config.sidecar_log_path),
        }

    def _ensure_binary(self) -> Path:
        path = self.config.sidecar_path
        if path.exists() and os.access(path, os.X_OK):
            return path
        if not self.config.auto_download:
            raise SidecarError(
                "cortexdb-grpc is not installed at " + str(path)
            )
        return self.download()

    def download(self) -> Path:
        last_error: Exception | None = None
        for asset in self._asset_names():
            try:
                data = self._fetch(asset)
                checksum = self._fetch_checksum(asset)
                actual = hashlib.sha256(data).hexdigest()
                if actual.lower() != checksum.lower():
                    raise SidecarError(
                        "checksum mismatch for " + asset
                    )
                binary = self._extract(data, asset)
                target = self.config.sidecar_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=str(target.parent), delete=False
                ) as handle:
                    handle.write(binary)
                    temp_path = Path(handle.name)
                if os.name != "nt":
                    temp_path.chmod(0o755)
                os.replace(str(temp_path), str(target))
                return target
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_error = exc
        raise SidecarError(
            "could not download cortexdb-grpc"
            + (": " + str(last_error) if last_error else "")
        )

    def _fetch(self, asset: str) -> bytes:
        url = "{}/v{}/{}".format(
            self.config.download_base_url.rstrip("/"),
            self.config.sidecar_version.lstrip("v"),
            asset,
        )
        with urllib.request.urlopen(
            url, timeout=60, context=ssl_context()
        ) as response:
            return response.read()

    def _fetch_checksum(self, asset: str) -> str:
        url = "{}/v{}/{}.sha256".format(
            self.config.download_base_url.rstrip("/"),
            self.config.sidecar_version.lstrip("v"),
            asset,
        )
        with urllib.request.urlopen(
            url, timeout=30, context=ssl_context()
        ) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
        if not text:
            raise SidecarError("empty checksum for " + asset)
        return text.split()[0]

    def _asset_names(self) -> list[str]:
        system = _goos()
        arch = _goarch()
        names = [f"cortexdb-grpc_{system}_{arch}.tar.gz"]
        if system == "windows":
            names.insert(0, f"cortexdb-grpc_{system}_{arch}.zip")
        return names

    def _extract(self, data: bytes, asset: str) -> bytes:
        if asset.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    name = PurePosixPath(member.filename)
                    if name.name != "cortexdb-grpc" and name.name != "cortexdb-grpc.exe":
                        continue
                    return archive.read(member)
            raise SidecarError("archive does not contain cortexdb-grpc")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts:
                    continue
                if name.name not in {"cortexdb-grpc", "cortexdb-grpc.exe"}:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                return extracted.read()
        raise SidecarError("archive does not contain cortexdb-grpc")

    def _load_or_create_token(self) -> str:
        path = self.config.token_path
        try:
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        except OSError:
            pass
        token = secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return token

    def _child_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CORTEXDB_EMBED")
            and key not in {
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "CORTEXDB_KEY_FILE",
            }
        }
        env.update(
            {
                "CORTEXDB_PATH": str(self.config.db_path),
                "CORTEXDB_HTTP_ADDR": self._http_addr,
                "CORTEXDB_GRPC_ADDR": self._grpc_addr,
                "CORTEXDB_GRPC_TOKEN": self._token,
                "CORTEXDB_BACKUP_DIR": str(self.config.backup_dir),
            }
        )
        return env

    def _resolve_addr(self, address: str) -> str:
        host, _, port_text = address.rpartition(":")
        if not host:
            host, port_text = "127.0.0.1", address
        try:
            port = int(port_text)
        except ValueError:
            port = 0
        if port == 0:
            port = _free_port(host)
        return f"{host}:{port}"

    def _log_tail(self, limit: int = 2000) -> str:
        try:
            data = self.config.sidecar_log_path.read_bytes()
        except OSError:
            return ""
        return data[-limit:].decode("utf-8", errors="replace")


def _goos() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win"):
        return "windows"
    raise SidecarError("unsupported platform: " + sys.platform)


def _goarch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine in {"x86_64", "amd64", "x64"}:
        return "amd64"
    raise SidecarError("unsupported architecture: " + machine)


def _free_port(host: str) -> int:
    bind_host = host if host not in {"", "0.0.0.0"} else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])
