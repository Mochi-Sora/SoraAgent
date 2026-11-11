"""Download, start, and stop the loopback CortexDB sidecar."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import secrets
import signal
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
from urllib.parse import urlsplit

from .client import CortexDBClient
from .config import MemoryConfig
from .net import ssl_context

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Only variables the sidecar genuinely needs.  Everything else, including
# unrelated provider keys, is withheld from the child process.
_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
        "WINDIR",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "COMSPEC",
        "PATHEXT",
        "OS",
        "NUMBER_OF_PROCESSORS",
    }
)


def _is_loopback(host: str | None) -> bool:
    value = (host or "").strip("[]").lower()
    return value in _LOOPBACK_HOSTS or value.startswith("127.")


def _check_download_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise SidecarError(
            "sidecar download URL must start with http:// or https://"
        )
    if parsed.username or parsed.password:
        raise SidecarError(
            "sidecar download URL must not contain credentials"
        )
    if parsed.scheme != "https" and not _is_loopback(parsed.hostname):
        raise SidecarError(
            "refusing to download the sidecar over plain HTTP from "
            f"{parsed.hostname or url!r}; use https:// or a loopback address"
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_download_url(newurl)
        return super().redirect_request(
            req, fp, code, msg, headers, newurl
        )


class SidecarError(RuntimeError):
    pass


class SidecarManager:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._client: CortexDBClient | None = None
        self._http_addr = ""
        self._grpc_addr = ""
        self._token = ""
        self._lock_handle: Any = None
        # True when this manager connected to a sidecar started by another
        # process.  Closing an adopted sidecar must not kill the owner's child.
        self._adopted = False
        self._adopted_pid = 0

    @property
    def client(self) -> CortexDBClient | None:
        return self._client

    def ensure_running(self) -> CortexDBClient:
        if self._client is not None and self._healthy(self._client):
            return self._client
        self.config.ensure_directories()
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._client = None
        if self._client is not None:
            if self._healthy(self._client):
                return self._client
            self._client = None
            if self._process is not None and self._process.poll() is None:
                self.stop()
        if self._adopt_running():
            assert self._client is not None
            return self._client
        try:
            self._acquire_lock()
        except SidecarError:
            # Another process may be starting the sidecar right now.  Give it
            # a moment to publish its endpoint before giving up.
            if self._wait_for_adoption():
                assert self._client is not None
                return self._client
            raise
        try:
            binary = self._ensure_binary()
            self._token = self.config.token or self._load_or_create_token()
            self._http_addr = self._resolve_addr(self.config.http_addr)
            self._grpc_addr = self._resolve_addr(self.config.grpc_addr)
            self._adopted = False
            self._adopted_pid = 0
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
        except Exception:
            self._release_lock()
            raise
        client = CortexDBClient(
            self._http_addr, self._token, timeout=self.config.request_timeout
        )
        self._client = client
        deadline = time.monotonic() + max(1.0, self.config.sidecar_start_timeout)
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                message = self._log_tail()
                self.stop()
                raise SidecarError(
                    "cortexdb-grpc exited during startup"
                    + (": " + message if message else "")
                )
            if client.health():
                self._write_state()
                return client
            time.sleep(0.1)
        self.stop()
        raise SidecarError(
            "cortexdb-grpc did not become healthy at " + self._http_addr
        )

    def stop(self, *, force: bool = False) -> None:
        process = self._process
        self._process = None
        self._client = None
        adopted_pid = self._adopted_pid if self._adopted else 0
        self._adopted = False
        self._adopted_pid = 0
        if process is None or process.poll() is not None:
            self._release_lock()
            if force:
                if adopted_pid:
                    self._terminate_pid(adopted_pid)
                self._clear_state()
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        except OSError:
            pass
        finally:
            self._release_lock()
            self._clear_state()

    def health(self) -> bool:
        if self._client is None:
            return False
        return self._healthy(self._client)

    def doctor(self) -> dict[str, Any]:
        if self._client is None:
            self._adopt_running()
        binary = self.config.sidecar_path
        running = self._process is not None and self._process.poll() is None
        if not running and self._adopted and self._client is not None:
            running = self._healthy(self._client)
        return {
            "enabled": self.config.enabled,
            "binary": str(binary),
            "binary_present": binary.exists(),
            "binary_verified": self._binary_verified(binary),
            "running": running,
            "healthy": self.health(),
            "http_addr": self._http_addr,
            "grpc_addr": self._grpc_addr,
            "db_path": str(self.config.db_path),
            "backup_dir": str(self.config.backup_dir),
            "version": self.config.sidecar_version,
            "log": str(self.config.sidecar_log_path),
        }

    # -- endpoint discovery ----------------------------------------------------

    @staticmethod
    def _healthy(client: CortexDBClient) -> bool:
        try:
            return bool(client.health())
        except Exception:  # noqa: BLE001 - a broken probe is a dead sidecar
            return False

    def _read_token(self) -> str:
        if self.config.token:
            return self.config.token
        try:
            return self.config.token_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _read_state(self) -> dict[str, Any] | None:
        try:
            raw = json.loads(
                self.config.sidecar_state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        if str(raw.get("db_path") or "") != str(self.config.db_path):
            return None
        http_addr = str(raw.get("http_addr") or "").strip()
        if not http_addr:
            return None
        return {
            "http_addr": http_addr,
            "grpc_addr": str(raw.get("grpc_addr") or "").strip(),
            "pid": int(raw.get("pid") or 0),
        }

    def _write_state(self) -> None:
        if not self._http_addr:
            return
        payload = {
            "version": self.config.sidecar_version,
            "db_path": str(self.config.db_path),
            "http_addr": self._http_addr,
            "grpc_addr": self._grpc_addr,
            "pid": self._adopted_pid
            or (self._process.pid if self._process is not None else 0),
            "updated_at": time.time(),
        }
        target = self.config.sidecar_state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(target.parent),
                delete=False,
            ) as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                temporary = Path(handle.name)
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            os.replace(str(temporary), str(target))
        except OSError:
            pass

    def _clear_state(self) -> None:
        try:
            self.config.sidecar_state_path.unlink()
        except OSError:
            pass

    def _adopt_running(self) -> bool:
        candidates: list[dict[str, Any]] = []
        state = self._read_state()
        if state is not None:
            token = self._read_token()
            if token:
                candidates.append({**state, "token": token})
            elif state.get("pid") and not self._pid_alive(int(state["pid"])):
                self._clear_state()
        candidates.extend(self._linux_candidates())
        for candidate in candidates:
            http_addr = str(candidate.get("http_addr") or "").strip()
            token = str(candidate.get("token") or "").strip()
            if not http_addr or not token:
                continue
            client = CortexDBClient(
                http_addr, token, timeout=self.config.request_timeout
            )
            if not self._healthy(client):
                continue
            self._client = client
            self._token = token
            self._http_addr = http_addr
            self._grpc_addr = str(candidate.get("grpc_addr") or "").strip()
            self._adopted = True
            self._adopted_pid = int(candidate.get("pid") or 0)
            self._write_state()
            return True
        return False

    def _wait_for_adoption(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if self._adopt_running():
                return True
            time.sleep(0.1)
        return False

    def _linux_candidates(
        self, proc_root: Path | None = None
    ) -> list[dict[str, Any]]:
        if proc_root is None and sys.platform != "linux":
            return []
        root = proc_root or Path("/proc")
        target = str(self.config.db_path)
        out: list[dict[str, Any]] = []
        try:
            entries = list(root.iterdir())
        except OSError:
            return out
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                comm = (entry / "comm").read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except OSError:
                continue
            if comm not in {"cortexdb-grpc", "cortexdb-grpc.exe"}:
                continue
            try:
                raw = (entry / "environ").read_bytes()
            except OSError:
                continue
            env: dict[str, str] = {}
            for item in raw.split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                env[key.decode("utf-8", "replace")] = value.decode(
                    "utf-8", "replace"
                )
            if env.get("CORTEXDB_PATH") != target:
                continue
            http_addr = env.get("CORTEXDB_HTTP_ADDR", "").strip()
            token = env.get("CORTEXDB_GRPC_TOKEN", "").strip()
            if not http_addr or not token:
                continue
            out.append(
                {
                    "http_addr": http_addr,
                    "grpc_addr": env.get("CORTEXDB_GRPC_ADDR", "").strip(),
                    "token": token,
                    "pid": int(entry.name),
                }
            )
        return out

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _terminate_pid(pid: int) -> None:
        if pid <= 0 or not SidecarManager._is_sidecar_pid(pid):
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not SidecarManager._pid_alive(pid):
                return
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    @staticmethod
    def _is_sidecar_pid(pid: int) -> bool:
        if sys.platform != "linux":
            return True
        try:
            comm = Path(f"/proc/{pid}/comm").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            return False
        return comm in {"cortexdb-grpc", "cortexdb-grpc.exe"}

    def _binary_verified(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            self._verify_binary(path)
        except SidecarError:
            return False
        return True

    def _ensure_binary(self) -> Path:
        path = self.config.sidecar_path
        if path.exists() and os.access(path, os.X_OK):
            self._verify_binary(path)
            return path
        if not self.config.auto_download:
            raise SidecarError(
                "cortexdb-grpc is not installed at " + str(path)
            )
        return self.download()

    def _checksum_path(self, binary: Path) -> Path:
        return binary.with_name(binary.name + ".sha256")

    def _read_pin(self, binary: Path) -> str:
        try:
            payload = json.loads(
                self._checksum_path(binary).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return ""
        if not isinstance(payload, dict):
            return ""
        if str(payload.get("version", "")) != self.config.sidecar_version:
            return ""
        value = str(payload.get("sha256", "")).strip().lower()
        return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""

    def _pinned_checksum(self) -> str:
        value = (self.config.sidecar_sha256 or "").strip().lower()
        if not value:
            return ""
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise SidecarError(
                "SORA_MEMORY_SIDECAR_SHA256 must be 64 hexadecimal characters"
            )
        return value

    def _verify_binary(self, path: Path) -> None:
        expected = self._pinned_checksum() or self._read_pin(path)
        if not expected:
            raise SidecarError(
                "refusing to run unverified cortexdb-grpc at "
                f"{path}; re-download it or set "
                "SORA_MEMORY_SIDECAR_SHA256"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SidecarError(
                "cortexdb-grpc checksum mismatch at " + str(path)
            )

    def download(self) -> Path:
        pinned = self._pinned_checksum() or self._read_pin(
            self.config.sidecar_path
        )
        last_error: Exception | None = None
        for asset in self._asset_names():
            try:
                data = self._fetch(asset)
                checksum = pinned or self._fetch_checksum(asset)
                checksum = checksum.strip().lower()
                if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                    raise SidecarError(
                        "invalid checksum for " + asset
                    )
                actual = hashlib.sha256(data).hexdigest()
                if actual != checksum:
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
                self._write_checksum(target, checksum)
                return target
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_error = exc
        raise SidecarError(
            "could not download cortexdb-grpc"
            + (": " + str(last_error) if last_error else "")
        )

    def _write_checksum(self, binary: Path, checksum: str) -> None:
        target = self._checksum_path(binary)
        payload = json.dumps(
            {
                "version": self.config.sidecar_version,
                "sha256": checksum,
            },
            sort_keys=True,
        )
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(target.parent),
            delete=False,
        ) as handle:
            handle.write(payload + "\n")
            temporary = Path(handle.name)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(str(temporary), str(target))

    def _fetch(self, asset: str) -> bytes:
        url = "{}/v{}/{}".format(
            self.config.download_base_url.rstrip("/"),
            self.config.sidecar_version.lstrip("v"),
            asset,
        )
        with self._open(url, timeout=60) as response:
            return response.read()

    def _fetch_checksum(self, asset: str) -> str:
        url = "{}/v{}/{}.sha256".format(
            self.config.download_base_url.rstrip("/"),
            self.config.sidecar_version.lstrip("v"),
            asset,
        )
        with self._open(url, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
        if not text:
            raise SidecarError("empty checksum for " + asset)
        return text.split()[0]

    @staticmethod
    def _open(url: str, timeout: float) -> Any:
        _check_download_url(url)
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context()),
            _SafeRedirectHandler(),
        )
        return opener.open(url, timeout=timeout)

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
            if key in _CHILD_ENV_ALLOWLIST
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

    def _acquire_lock(self) -> None:
        if self._lock_handle is not None:
            return
        path = self.config.memory_dir / "sidecar.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+")  # noqa: SIM115 - held while the child runs
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise SidecarError(
                "another SoraAgent process is already managing the "
                "cortexdb sidecar: " + str(path)
            ) from exc
        self._lock_handle = handle

    def _release_lock(self) -> None:
        handle = self._lock_handle
        self._lock_handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            handle.close()

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
