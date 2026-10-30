from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sora_agent.agent import Agent, AgentResult
from sora_agent.identity import IdentityConfig, IdentityStore
from sora_agent.memory.config import MemoryConfig
from sora_agent.memory.daily import DailyStore
from sora_agent.memory.hook import MemoryHook
from sora_agent.memory.recall import MemoryRecallService
from sora_agent.providers import Message


class ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


class FakeBackend:
    def __init__(self) -> None:
        self.memories: dict[str, dict[str, Any]] = {}
        self.recall_results: list[dict[str, Any]] = []

    def health(self) -> bool:
        return True

    def save_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self.memories.get(memory_id)

    def list_memories(self, limit: int = 500) -> list[dict[str, Any]]:
        return list(self.memories.values())[:limit]

    def update_memory(self, memory_id: str, **fields: Any) -> dict[str, Any]:
        record = self.memories[memory_id]
        for key, value in fields.items():
            if key == "metadata":
                record.setdefault("metadata", {}).update(value or {})
            else:
                record[key] = value
        return record

    def delete_memory(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)

    def recall(self, query: str, top_k: int = 8, **options: Any) -> dict[str, Any]:
        return {"memories": list(self.recall_results)}

    def rules_apply(self, **payload: Any) -> dict[str, Any]:
        return {"derived": 0}

    def sweep_expired(self) -> dict[str, Any]:
        return {"swept": 0}


def config_for(root: Path) -> MemoryConfig:
    return MemoryConfig(root=root, auto_download=False)


def profile_store(tmp_path: Path) -> IdentityStore:
    store = IdentityStore(IdentityConfig(root=tmp_path / "identity"))
    store.ensure("user")
    assert store.apply(
        "user",
        "- Name: Jaden\n- Age: 24\n- Likes: potatoes\n",
    ).status == "accepted"
    return store


def assert_no_profile_facts(text: str) -> None:
    for secret in ("Jaden", "24", "potatoes"):
        assert secret not in text


def test_day_buffer_is_redacted_against_the_profile(tmp_path: Path) -> None:
    store = profile_store(tmp_path)
    hook = MemoryHook(
        config_for(tmp_path),
        client=FakeBackend(),
        identity=store,
    )
    task = "I'm Jaden, 24, and I like potatoes."

    hook.before_run(task)
    assert hook.daily.count() == 0

    hook.after_run(SimpleNamespace(proposal="The task is complete."))
    today = datetime.now(timezone.utc).date()
    rows = hook.daily.read_day(today)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert_no_profile_facts(rows[0]["content"])
    assert "task is complete" in rows[1]["content"]


def test_identity_edit_applies_before_the_user_message_is_stored(
    tmp_path: Path,
) -> None:
    store = IdentityStore(IdentityConfig(root=tmp_path / "identity"))
    hook = MemoryHook(
        config_for(tmp_path),
        client=FakeBackend(),
        identity=store,
    )
    provider = ScriptedProvider(
        [
            "Got it.\n"
            "<user_edit>\n"
            "- Name: Jaden\n- Age: 24\n- Likes: potatoes\n"
            "</user_edit>"
        ]
    )
    agent = Agent(provider=provider, memory=hook, identity=store)

    result = agent.run("I'm Jaden, 24, and I like potatoes.")
    assert result.proposal == "Got it."
    assert "- Name: Jaden" in store.load("user").content

    rows = hook.daily.read_day(datetime.now(timezone.utc).date())
    assert [row["role"] for row in rows] == ["user", "assistant"]
    for row in rows:
        assert_no_profile_facts(row["content"])


def test_memory_persists_only_the_final_answer(tmp_path: Path) -> None:
    hook = MemoryHook(
        config_for(tmp_path),
        client=FakeBackend(),
    )

    hook.before_run("Check PO-1234.")
    hook.after_run(
        SimpleNamespace(
            final_answer="PO-1234 is open.",
            proposal='ACTION: lookup_order(order_id="PO-1234")',
        )
    )

    rows = hook.daily.read_day(datetime.now(timezone.utc).date())
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[1]["content"] == "PO-1234 is open."
    assert "ACTION:" not in rows[1]["content"]


def test_memory_does_not_store_intermediate_action_scaffolding(
    tmp_path: Path,
) -> None:
    hook = MemoryHook(
        config_for(tmp_path),
        client=FakeBackend(),
    )

    hook.before_run("Check PO-1234.")
    hook.after_run(
        AgentResult(
            task="Check PO-1234.",
            proposal='ACTION: lookup_order(order_id="PO-1234")',
            stop_reason="max_steps",
        )
    )

    rows = hook.daily.read_day(datetime.now(timezone.utc).date())
    assert [row["role"] for row in rows] == ["user"]


def test_dream_redacts_legacy_daily_content_before_transfer(
    tmp_path: Path,
) -> None:
    store = profile_store(tmp_path)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    daily = DailyStore(
        tmp_path / "memory" / "daily",
        clock=lambda: yesterday,
    )
    daily.append("user", "I'm Jaden, 24, and I like potatoes.")
    backend = FakeBackend()
    hook = MemoryHook(
        config_for(tmp_path),
        client=backend,
        daily=daily,
        identity=store,
    )

    report = hook.dream()
    assert report["errors"] == []
    assert len(backend.memories) == 1
    record = next(iter(backend.memories.values()))
    assert_no_profile_facts(record["content"])
    assert record["metadata"]["profile_redacted"] is True
    assert not daily.path_for(yesterday.date()).exists()


def test_dream_skips_fully_redacted_daily_rows(tmp_path: Path) -> None:
    store = profile_store(tmp_path)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    daily = DailyStore(
        tmp_path / "memory" / "daily",
        clock=lambda: yesterday,
    )
    daily.append(
        "user",
        store.redact_for_memory(
            "<user_edit>\n- Name: Jaden\n</user_edit>"
        ),
    )
    daily.append(
        "user",
        "<user_edit>\n- Name: Jaden\n</user_edit>",
    )
    backend = FakeBackend()
    hook = MemoryHook(
        config_for(tmp_path),
        client=backend,
        daily=daily,
        identity=store,
    )

    report = hook.dream()
    assert report["errors"] == []
    assert backend.memories == {}
    assert report["transferred_days"][0]["messages"] == 0
    assert not daily.path_for(yesterday.date()).exists()


def test_recall_redacts_daily_and_longterm_profile_content(
    tmp_path: Path,
) -> None:
    store = profile_store(tmp_path)
    fixed = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    daily = DailyStore(tmp_path / "daily", clock=lambda: fixed)
    daily.append(
        "user",
        "Jaden likes potatoes and the deploy is Monday.",
    )
    backend = FakeBackend()
    backend.recall_results = [
        {
            "memory": {
                "id": "profile-only",
                "content": "<user_edit>\n- Name: Jaden\n</user_edit>",
            },
            "score": 0.9,
        },
        {
            "memory": {
                "id": "deploy",
                "content": "The deploy moved to Monday.",
            },
            "score": 0.8,
        },
    ]
    service = MemoryRecallService(
        daily,
        lambda: backend,
        redactor=store.redact_for_memory,
    )

    text = service.recall("deploy")
    assert "deploy is Monday" in text
    assert "The deploy moved to Monday." in text
    assert "profile-only" not in text
    assert_no_profile_facts(text)


def test_recall_filters_out_fully_redacted_hits(tmp_path: Path) -> None:
    store = profile_store(tmp_path)
    fixed = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    daily = DailyStore(tmp_path / "daily", clock=lambda: fixed)
    daily.append(
        "user",
        "<user_edit>\n- Name: Jaden\n</user_edit>",
    )
    backend = FakeBackend()
    backend.recall_results = [
        {
            "memory": {
                "id": "profile-only",
                "content": "<user_edit>\n- Name: Jaden\n</user_edit>",
            },
            "score": 0.9,
        }
    ]
    service = MemoryRecallService(
        daily,
        lambda: backend,
        redactor=store.redact_for_memory,
    )

    daily_items, _ = service.collect("Name", scope="daily")
    longterm_items, _ = service.collect("Name", scope="longterm")
    assert daily_items == []
    assert longterm_items == []


def test_redactor_failure_drops_the_memory_write(tmp_path: Path) -> None:
    class BrokenIdentity:
        def redact_for_memory(self, text: str, source: str = "user") -> str:
            raise RuntimeError("redactor down")

    hook = MemoryHook(
        config_for(tmp_path),
        client=FakeBackend(),
        identity=BrokenIdentity(),
    )
    hook.before_run("sensitive task")
    hook.after_run(SimpleNamespace(proposal="done"))
    assert hook.daily.count() == 0
