"""Compressed, verifiable snapshots of the active and daily databases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ._zstd import available as zstd_available
from ._zstd import compress_file, decompress_file
from .config import MemoryConfig


class BackupError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip(".-")
    return cleaned or "snapshot"


class BackupManager:
    def __init__(
        self,
        config: MemoryConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        config.ensure_directories()
        try:
            config.backup_dir.chmod(0o700)
        except OSError:
            pass

    def now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def snapshot_active(self, source: Path | None = None) -> dict[str, Any]:
        return self.snapshot(source or self.config.db_path, "active", "active")

    def snapshot_daily(
        self, source: Path, day: date | None = None
    ) -> dict[str, Any]:
        label = (day or date.fromisoformat(Path(source).stem)).isoformat()
        return self.snapshot(source, "daily", label)

    def snapshot(self, source: Path, kind: str, label: str) -> dict[str, Any]:
        if not zstd_available():
            raise BackupError(
                "memory snapshots require zstd support: install the optional "
                "'zstandard' package (or use Python 3.14+)"
            )
        source = Path(source)
        if not source.exists():
            raise BackupError("database does not exist: " + str(source))
        stamp = self.now().strftime("%Y-%m-%dT%H%M%S%fZ")
        filename = f"{stamp}-{_safe_label(kind)}-{_safe_label(label)}.db.zst"
        target = self.config.backup_dir / filename
        if target.exists():
            filename = f"{stamp}-{_safe_label(kind)}-{_safe_label(label)}-{os.getpid()}.db.zst"
            target = self.config.backup_dir / filename
        with tempfile.TemporaryDirectory(
            prefix=".snapshot-", dir=str(self.config.backup_dir)
        ) as tmp:
            raw = Path(tmp) / "snapshot.db"
            self._vacuum_into(source, raw)
            self._integrity_check(raw)
            uncompressed_hash = _sha256(raw)
            compress_file(raw, target)
        try:
            target.chmod(0o600)
        except OSError:
            pass
        compressed_hash = _sha256(target)
        manifest = {
            "version": 1,
            "file": filename,
            "kind": kind,
            "label": _safe_label(label),
            "created_at": self.now().isoformat(),
            "source": str(source),
            "sha256": compressed_hash,
            "uncompressed_sha256": uncompressed_hash,
            "bytes": target.stat().st_size,
        }
        self._write_json(Path(str(target) + ".json"), manifest)
        self._update_latest(manifest)
        self.prune()
        return manifest

    def restore(
        self,
        snapshot: Path,
        target: Path | None = None,
        stop: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if not zstd_available():
            raise BackupError(
                "restoring a memory snapshot requires zstd support: install "
                "the optional 'zstandard' package (or use Python 3.14+)"
            )
        snapshot = Path(snapshot)
        if not snapshot.exists():
            raise BackupError("snapshot does not exist: " + str(snapshot))
        manifest = self._manifest_for(snapshot)
        expected = str(manifest.get("sha256", ""))
        actual = _sha256(snapshot)
        if not expected or actual != expected:
            raise BackupError("snapshot checksum mismatch")
        target = Path(target or self.config.db_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".restore-", dir=str(target.parent)
        ) as tmp:
            restored = Path(tmp) / "restored.db"
            decompress_file(snapshot, restored)
            self._integrity_check(restored)
            if stop is not None:
                stop()
            for suffix in ("-wal", "-shm"):
                stale = Path(str(target) + suffix)
                if stale.exists():
                    stale.unlink()
            previous = None
            if target.exists():
                previous = target.with_name(
                    target.name
                    + ".pre-restore-"
                    + self.now().strftime("%Y%m%dT%H%M%S%fZ")
                )
                os.replace(str(target), str(previous))
            os.replace(str(restored), str(target))
        try:
            target.chmod(0o600)
        except OSError:
            pass
        return {
            "restored": str(target),
            "snapshot": str(snapshot),
            "previous": str(previous) if previous else "",
            "integrity": "ok",
        }

    def prune(self) -> list[Path]:
        manifests = self.list_snapshots()
        if not manifests:
            return []
        keep: set[str] = set()
        active = [m for m in manifests if m.get("kind") == "active"]
        for manifest in active[-max(1, self.config.backup_keep_active):]:
            keep.add(manifest["file"])
        daily = [m for m in manifests if m.get("kind") == "daily"]
        daily.sort(key=lambda item: str(item.get("created_at", "")))
        by_day: dict[str, dict[str, Any]] = {}
        for manifest in daily:
            day = _manifest_day(manifest)
            if day:
                by_day[day] = manifest
        ordered_days = sorted(by_day)
        for day in ordered_days[-max(1, self.config.backup_keep_daily):]:
            keep.add(by_day[day]["file"])
        weeks: dict[str, dict[str, Any]] = {}
        months: dict[str, dict[str, Any]] = {}
        for day in ordered_days:
            try:
                parsed = date.fromisoformat(day)
            except ValueError:
                continue
            iso = parsed.isocalendar()
            weeks[f"{iso[0]}-{iso[1]:02d}"] = by_day[day]
            months[day[:7]] = by_day[day]
        for key in sorted(weeks)[-max(1, self.config.backup_keep_weekly):]:
            keep.add(weeks[key]["file"])
        for key in sorted(months)[-max(1, self.config.backup_keep_monthly):]:
            keep.add(months[key]["file"])
        removed: list[Path] = []
        for manifest in manifests:
            if manifest["file"] in keep:
                continue
            path = self._safe_snapshot_path(str(manifest["file"]))
            if path is None:
                continue
            for candidate in (path, Path(str(path) + ".json")):
                try:
                    candidate.unlink()
                    removed.append(candidate)
                except FileNotFoundError:
                    pass
        self._refresh_latest()
        return removed

    def _safe_snapshot_path(self, filename: str) -> Path | None:
        """Return a path inside the backup directory, or ``None``.

        Manifests are JSON files that a local attacker or a corrupt write can
        modify.  Pruning must never follow a name such as ``../../.ssh`` out
        of the backup directory or through a symlink.
        """
        name = str(filename).strip()
        if not name or Path(name).name != name:
            return None
        if "/" in name or "\\" in name or name in {".", ".."}:
            return None
        path = self.config.backup_dir / name
        root = self.config.backup_dir.resolve()
        try:
            path.resolve().relative_to(root)
        except (OSError, ValueError):
            return None
        return path

    def list_snapshots(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        for path in sorted(self.config.backup_dir.glob("*.db.zst.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("file"):
                manifests.append(data)
        manifests.sort(key=lambda item: str(item.get("created_at", "")))
        return manifests

    def latest(self, kind: str = "active") -> dict[str, Any] | None:
        for manifest in reversed(self.list_snapshots()):
            if manifest.get("kind") == kind:
                return manifest
        return None

    def stats(self) -> dict[str, Any]:
        manifests = self.list_snapshots()
        return {
            "count": len(manifests),
            "active": sum(1 for item in manifests if item.get("kind") == "active"),
            "daily": sum(1 for item in manifests if item.get("kind") == "daily"),
            "bytes": sum(int(item.get("bytes", 0)) for item in manifests),
            "latest": manifests[-1]["file"] if manifests else "",
        }

    def _vacuum_into(self, source: Path, target: Path) -> None:
        if target.exists():
            target.unlink()
        conn = sqlite3.connect(str(source), timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            try:
                conn.execute("VACUUM INTO ?", (str(target),))
            except sqlite3.OperationalError:
                escaped = str(target).replace("'", "''")
                conn.execute("VACUUM INTO '" + escaped + "'")
        finally:
            conn.close()

    @staticmethod
    def _integrity_check(path: Path) -> None:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
        if not row or str(row[0]).lower() != "ok":
            raise BackupError("SQLite integrity check failed: " + str(path))

    def _manifest_for(self, snapshot: Path) -> dict[str, Any]:
        path = Path(str(snapshot) + ".json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackupError("snapshot manifest is missing or invalid") from exc
        if not isinstance(data, dict):
            raise BackupError("snapshot manifest is invalid")
        return data

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        os.replace(str(temp), str(path))

    def _update_latest(self, manifest: dict[str, Any]) -> None:
        path = self.config.backup_dir / "latest.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[str(manifest.get("kind", "active"))] = manifest
        self._write_json(path, data)

    def _refresh_latest(self) -> None:
        path = self.config.backup_dir / "latest.json"
        current: dict[str, Any] = {}
        for kind in ("active", "daily"):
            manifest = self.latest(kind)
            if manifest:
                current[kind] = manifest
        if current:
            self._write_json(path, current)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _manifest_day(manifest: dict[str, Any]) -> str:
    label = str(manifest.get("label", ""))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", label):
        return label
    created = str(manifest.get("created_at", ""))
    return created[:10] if len(created) >= 10 else ""
