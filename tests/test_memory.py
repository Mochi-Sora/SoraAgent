from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
import threading
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from sora_agent.agent import Agent
from sora_agent.memory.backup import BackupError, BackupManager
from sora_agent.memory.client import CortexDBClient, CortexDBError
from sora_agent.memory.config import MemoryConfig
from sora_agent.memory.daily import DailyStore
from sora_agent.memory.dream import DreamRunner
from sora_agent.memory.gate import RecallEnforcingProvider
from sora_agent.memory.hook import MemoryHook
from sora_agent.memory.louvain import louvain_communities
from sora_agent.memory.recall import MemoryRecallService
from sora_agent.memory.sidecar import SidecarError, SidecarManager
from sora_agent.providers import Message
from sora_agent.tools import Tool


class TextProvider:
    def __init__(self, text: str = "The task is complete.") -> None:
        self.text = text
        self.calls: List[List[Message]] = []

    def complete(self, messages, tools):
        self.calls.append(list(messages))
        return self.text


class ScriptedProvider:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[List[Any]] = []

    def complete(self, messages, tools):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class FakeBackend:
    def __init__(self) -> None:
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.fail_save = False

    def health(self) -> bool:
        return True

    def save_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_save:
            raise RuntimeError("save failed")
        memory_id = str(payload["memory_id"])
        record = {
            "id": memory_id,
            "content": payload.get("content", ""),
            "metadata": dict(payload.get("metadata") or {}),
            "importance": float(payload.get("importance") or 0.0),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.memories[memory_id] = record
        return {"memory": record}

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        return self.memories.get(memory_id)

    def list_memories(self, limit: int = 500) -> List[Dict[str, Any]]:
        return list(self.memories.values())[:limit]

    def update_memory(self, memory_id: str, **fields: Any) -> Dict[str, Any]:
        record = self.memories[memory_id]
        for key, value in fields.items():
            if key == "metadata":
                record.setdefault("metadata", {}).update(value or {})
            else:
                record[key] = value
        return record

    def delete_memory(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)

    def recall(self, query: str, top_k: int = 8, **options: Any) -> Dict[str, Any]:
        return {"memories": []}

    def rules_apply(self, **payload: Any) -> Dict[str, Any]:
        return {"derived": 0}

    def sweep_expired(self) -> Dict[str, Any]:
        return {"swept": 0}


def config_for(root: Path, **kwargs: Any) -> MemoryConfig:
    values = {"root": root, "auto_download": False}
    values.update(kwargs)
    return MemoryConfig(**values)


def test_config_defaults_and_env(tmp_path: Path, monkeypatch) -> None:
    config = MemoryConfig(root=tmp_path)
    assert config.enabled is True
    assert config.backup_dir == tmp_path / "memory backup"
    assert config.daily_dir == tmp_path / "memory" / "daily"
    monkeypatch.setenv("SORA_MEMORY", "0")
    assert MemoryConfig.from_env().enabled is False
    monkeypatch.setenv("SORA_MEMORY", "1")
    monkeypatch.setenv("SORA_MEMORY_FORCE_RECALL", "1")
    loaded = MemoryConfig.from_env()
    assert loaded.enabled is True
    assert loaded.force_first_recall is True


def test_daily_store_is_verbatim_and_searchable(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    store = DailyStore(tmp_path, clock=lambda: moment)
    first = store.append("user", "Please deploy the release on Friday.")
    second = store.append("assistant", "The deploy is scheduled for Friday.")
    assert store.count() == 2
    with pytest.raises(ValueError):
        store.append("tool", "secret tool output")
    hits = store.search("deploy Friday")
    assert {hit.message_id for hit in hits} == {first, second}
    assert store.read_day(moment.date())[0]["role"] == "user"
    tomorrow = moment + timedelta(days=1)
    assert store.pending_days(tomorrow) == [moment.date()]
    assert store.search("deploy", days=2)  # yesterday is still searchable
    assert store.delete_day(moment.date()) is True
    assert not store.path_for(moment.date()).exists()


def test_daily_store_rejects_empty_content(tmp_path: Path) -> None:
    store = DailyStore(tmp_path)
    with pytest.raises(ValueError):
        store.append("user", "   ")


def test_backup_roundtrip_and_checksum(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config.ensure_directories()
    conn = sqlite3.connect(str(config.db_path))
    conn.execute("CREATE TABLE notes(value TEXT)")
    conn.execute("INSERT INTO notes VALUES ('hello')")
    conn.commit()
    conn.close()
    moment = datetime(2026, 10, 27, 1, tzinfo=timezone.utc)
    manager = BackupManager(config, clock=lambda: moment)
    manifest = manager.snapshot_active()
    snapshot = config.backup_dir / manifest["file"]
    assert snapshot.exists()
    assert Path(str(snapshot) + ".json").exists()
    target = tmp_path / "restored.db"
    Path(str(target) + "-wal").write_bytes(b"stale")
    Path(str(target) + "-shm").write_bytes(b"stale")
    result = manager.restore(snapshot, target=target)
    assert not Path(str(target) + "-wal").exists()
    assert not Path(str(target) + "-shm").exists()
    assert result["integrity"] == "ok"
    restored = sqlite3.connect(str(target))
    assert restored.execute("SELECT value FROM notes").fetchone()[0] == "hello"
    restored.close()
    original = snapshot.read_bytes()
    snapshot.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    with pytest.raises(BackupError):
        manager.restore(snapshot, target=tmp_path / "bad.db")


def test_backup_retention_keeps_daily_weekly_monthly(tmp_path: Path) -> None:
    config = config_for(
        tmp_path,
        backup_keep_daily=3,
        backup_keep_weekly=2,
        backup_keep_monthly=2,
        backup_keep_active=1,
    )
    config.ensure_directories()
    source = config.daily_dir / "source.db"
    conn = sqlite3.connect(str(source))
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()
    manager = BackupManager(config, clock=lambda: datetime(2026, 10, 27, tzinfo=timezone.utc))
    for offset in range(6):
        day = date(2026, 9, 1) + timedelta(days=offset * 7)
        manager.snapshot_daily(source, day)
    manager.prune()
    manifests = manager.list_snapshots()
    assert len(manifests) <= 5
    assert all((config.backup_dir / item["file"]).exists() for item in manifests)


class _ClientHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == "Bearer secret"

    def do_GET(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path == "/v1/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        self.server.requests.append(("DELETE", self.path, None))
        self._json(200, {"deleted": True})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        payload = json.loads(body or b"{}")
        self.server.requests.append(("POST", self.path, payload))
        if self.path == "/v1/memory":
            self._json(200, {"memory": payload, "created": True})
        elif self.path == "/v1/tools/memory_list_all":
            if payload.get("cursor") == "next":
                self._json(200, {"memories": [{"id": "m2"}], "truncated": False})
            else:
                self._json(
                    200,
                    {
                        "memories": [{"id": "m1"}],
                        "truncated": True,
                        "next_cursor": "next",
                    },
                )
        elif self.path == "/v1/tools/memory_update":
            self._json(200, {"memory": {"id": payload.get("memory_id")}})
        else:
            self._json(404, {"error": "not found"})


def _serve(handler) -> tuple:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_cortexdb_client_rest_and_pagination() -> None:
    server, thread = _serve(_ClientHandler)
    try:
        port = server.server_address[1]
        client = CortexDBClient("127.0.0.1:" + str(port), token="secret")
        assert client.health() is True
        assert client.save_memory({"memory_id": "m1", "content": "x"})["created"] is True
        memories = client.list_memories(limit=10)
        assert [item["id"] for item in memories] == ["m1", "m2"]
        client.update_memory("m1", importance=0.4)
        client.delete_memory("m1")
        with pytest.raises(CortexDBError):
            CortexDBClient("127.0.0.1:" + str(port), token="wrong").info()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_recall_merges_daily_and_longterm(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    daily = DailyStore(tmp_path, clock=lambda: moment)
    daily.append("user", "The deploy is on Friday.")

    class Backend(FakeBackend):
        def recall(self, query: str, top_k: int = 8, **options: Any) -> Dict[str, Any]:
            return {
                "memories": [
                    {
                        "memory": {"id": "long-1", "content": "The deploy moved to Monday."},
                        "score": 0.9,
                    }
                ]
            }

    service = MemoryRecallService(daily, lambda: Backend())
    text = service.recall("deploy")
    assert "daily:2026-10-26#1" in text
    assert "long-1" in text
    assert "untrusted" in text
    daily_only = service.recall("deploy", scope="daily")
    assert "long-1" not in daily_only


def test_recall_fails_open_without_backend(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    daily = DailyStore(tmp_path, clock=lambda: moment)
    daily.append("user", "The deploy is on Friday.")
    service = MemoryRecallService(daily, lambda: (_ for _ in ()).throw(RuntimeError("down")))
    text = service.recall("deploy")
    assert "Friday" in text
    assert "unavailable" in text


def test_forced_recall_gate_executes_tool_then_answers(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            'ACTION: memory_recall(query="deploy", scope="auto", k=3)',
            "The deploy is Monday.",
        ]
    )
    calls: List[Dict[str, Any]] = []

    def recall(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "Long-term memory: deploy moved to Monday."

    gate = RecallEnforcingProvider(provider, recall)
    result = gate.complete([Message("user", "When is the deploy?")], [])
    assert result == "The deploy is Monday."
    assert calls == [{"query": "deploy", "scope": "auto", "k": 3}]
    assert len(provider.calls) == 2


def test_forced_recall_gate_retries_then_fails_open() -> None:
    provider = ScriptedProvider(["I think Monday.", "Still Monday."])
    calls: List[str] = []
    gate = RecallEnforcingProvider(
        provider, lambda **kwargs: calls.append("called") or "memory", max_retries=1
    )
    result = gate.complete([Message("user", "When?")], [])
    assert result == "Still Monday."
    assert calls == []
    assert len(provider.calls) == 2


def test_louvain_separates_disjoint_clusters() -> None:
    edges = [
        ("a", "b", 1.0),
        ("b", "c", 1.0),
        ("a", "c", 1.0),
        ("d", "e", 1.0),
        ("e", "f", 1.0),
        ("d", "f", 1.0),
    ]
    communities = louvain_communities(edges)
    assert communities["a"] == communities["b"] == communities["c"]
    assert communities["d"] == communities["e"] == communities["f"]
    assert communities["a"] != communities["d"]


def _dream_setup(tmp_path: Path, now: datetime) -> tuple:
    config = config_for(tmp_path)
    config.ensure_directories()
    conn = sqlite3.connect(str(config.db_path))
    conn.execute("CREATE TABLE seed(x)")
    conn.commit()
    conn.close()
    daily = DailyStore(config.daily_dir, clock=lambda: now)
    backend = FakeBackend()
    return config, daily, backend


def test_dream_transfers_marks_and_deletes_daily(tmp_path: Path) -> None:
    day = datetime(2026, 10, 26, 23, tzinfo=timezone.utc)
    config, daily, backend = _dream_setup(tmp_path, day)
    daily.append("user", "Please deploy the release on Friday.")
    daily.append("assistant", "Understood, Friday it is.")
    now = datetime(2026, 10, 27, 0, 0, 1, tzinfo=timezone.utc)
    runner = DreamRunner(config, daily, backend, clock=lambda: now)
    report = runner.run()
    assert report["errors"] == []
    assert len(backend.memories) == 2
    assert not daily.path_for(day.date()).exists()
    for record in backend.memories.values():
        assert record["metadata"]["dream_round"] == 1
        assert record["metadata"]["dream_due"] == "2026-10-28T00:00:00+00:00"
    assert (config.backup_dir / "latest.json").exists()
    assert report["report_path"]
    second = DreamRunner(
        config, daily, backend, clock=lambda: datetime(2026, 10, 28, 0, 0, 1, tzinfo=timezone.utc)
    )
    second_report = second.run()
    assert second_report["errors"] == []
    for record in backend.memories.values():
        assert record["metadata"]["dream_round"] == 2
        assert record["metadata"]["dream_due"] is None


def test_dream_keeps_daily_when_transfer_fails(tmp_path: Path) -> None:
    day = datetime(2026, 10, 26, 23, tzinfo=timezone.utc)
    config, daily, backend = _dream_setup(tmp_path, day)
    daily.append("user", "Remember this.")
    backend.fail_save = True
    runner = DreamRunner(
        config, daily, backend, clock=lambda: datetime(2026, 10, 27, 0, 0, 1, tzinfo=timezone.utc)
    )
    report = runner.run()
    assert daily.path_for(day.date()).exists()
    assert report["errors"]
    assert report["transferred_days"] == []


def test_dream_creates_merge_and_conflict_records(tmp_path: Path) -> None:
    now = datetime(2026, 10, 27, 0, 0, 1, tzinfo=timezone.utc)
    config, daily, backend = _dream_setup(tmp_path, now)
    for memory_id, content in (
        ("a", "The release date is 2026-11-01 and the owner is Alice."),
        ("b", "The release date is 2026-11-01 and the owner is Alice."),
        ("c", "The release date is 2026-12-01 and the owner is Alice."),
    ):
        backend.save_memory(
            {
                "memory_id": memory_id,
                "content": content,
                "importance": 0.5,
                "metadata": {"sora_kind": "verbatim", "dream_round": 0},
            }
        )
    report = DreamRunner(config, daily, backend, clock=lambda: now).run()
    operations = {action["op"] for action in report["actions"]}
    assert "merge" in operations
    assert "conflict" in operations
    assert report["wake_requested"] is True
    assert any(record["metadata"].get("sora_kind") == "conflict" for record in backend.memories.values())


def test_tool_schema_supports_deferred_annotations() -> None:
    def lookup(order_id: str, limit: int = 5) -> str:
        return order_id

    schema = Tool(lookup, "lookup", "Look up an order.").schema
    assert schema["parameters"]["properties"] == {
        "order_id": {"type": "string"},
        "limit": {"type": "integer"},
    }
    assert schema["parameters"]["required"] == ["order_id"]


def test_memory_tool_schema(tmp_path: Path) -> None:
    hook = MemoryHook(config_for(tmp_path), client=FakeBackend())
    schema = hook.tool().schema
    assert schema["parameters"]["properties"]["query"] == {"type": "string"}
    assert schema["parameters"]["properties"]["k"] == {"type": "integer"}
    assert schema["parameters"]["required"] == ["query"]


def test_agent_hook_records_daily_messages(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    hook = MemoryHook(config, client=FakeBackend())
    agent = Agent(provider=TextProvider(), memory=hook)
    result = agent.run("Summarize the status.")
    assert result.proposal == "The task is complete."
    day = datetime.now(timezone.utc).date()
    assert hook.daily.count(day) == 2
    roles = [row["role"] for row in hook.daily.read_day(day)]
    assert roles == ["user", "assistant"]


def test_agent_voluntary_recall_feeds_result_back(tmp_path: Path) -> None:
    class Backend(FakeBackend):
        def recall(self, query: str, top_k: int = 8, **options: Any) -> Dict[str, Any]:
            return {
                "memories": [
                    {
                        "memory": {"id": "m1", "content": "The deploy is Monday."},
                        "score": 0.9,
                    }
                ]
            }

    config = config_for(tmp_path)
    hook = MemoryHook(config, client=Backend())
    provider = ScriptedProvider(
        [
            'ACTION: memory_recall(query="deploy")',
            "The deploy is Monday.",
        ]
    )
    agent = Agent(provider=provider, memory=hook)
    result = agent.run("When is the deploy?")
    assert result.proposal == "The deploy is Monday."
    tool_message = provider.calls[1][-1]
    assert tool_message.role == "tool"
    assert tool_message.name == "memory_recall"
    assert "The deploy is Monday." in tool_message.content


def test_agent_default_memory_is_opt_out(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SORA_MEMORY", "1")
    monkeypatch.setenv("SORA_MEMORY_ROOT", str(tmp_path))
    agent = Agent(provider=TextProvider())
    assert agent.memory is not None
    agent.run("hello")
    assert agent.memory.daily.count(datetime.now(timezone.utc).date()) == 2


def test_sidecar_extracts_binary_and_resolves_addresses(tmp_path: Path) -> None:
    manager = SidecarManager(config_for(tmp_path))
    assert any(name.endswith(".tar.gz") for name in manager._asset_names())
    address = manager._resolve_addr("127.0.0.1:0")
    assert not address.endswith(":0")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        binary = b"#!/bin/sh\necho fake\n"
        info = tarfile.TarInfo("cortexdb-grpc")
        info.size = len(binary)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(binary))
    assert manager._extract(data.getvalue(), "fake.tar.gz").startswith(b"#!")


class _DownloadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    asset = ""
    payload = b""
    checksum = ""

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.endswith(".sha256"):
            body = (self.checksum + "\n").encode()
        else:
            body = self.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_sidecar_download_verifies_checksum(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        binary = b"binary"
        info = tarfile.TarInfo("cortexdb-grpc")
        info.size = len(binary)
        archive.addfile(info, io.BytesIO(binary))
    data = payload.getvalue()
    _DownloadHandler.payload = data
    _DownloadHandler.checksum = hashlib.sha256(data).hexdigest()
    server, thread = _serve(_DownloadHandler)
    try:
        config = config_for(
            tmp_path,
            auto_download=True,
            sidecar_version="9.9.9",
            download_base_url="http://127.0.0.1:" + str(server.server_address[1]),
        )
        manager = SidecarManager(config)
        path = manager.download()
        assert path.exists()
        _DownloadHandler.checksum = "0" * 64
        with pytest.raises(SidecarError):
            SidecarManager(config).download()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
