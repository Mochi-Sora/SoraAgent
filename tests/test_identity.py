from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sora_agent.identity import (
    DEFAULT_RETENTION,
    PROFILE_WITHHELD,
    SOUL_LIMIT,
    USER_LIMIT,
    IdentityConfig,
    IdentityMonitor,
    IdentityStore,
    canonicalize,
    default_identity,
    extract_identity_edits,
)


class StepClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


def make_store(tmp_path: Path, **kwargs) -> IdentityStore:
    return IdentityStore(
        IdentityConfig(root=tmp_path / "identity", **kwargs),
        clock=StepClock(),
    )


def user_document(**overrides: str) -> str:
    values = {
        "Name": "Jaden",
        "Age": "24",
        "Likes": "potatoes",
    }
    values.update(overrides)
    return "".join(f"- {key}: {value}\n" for key, value in values.items())


def soul_document(voice: str = "dry") -> str:
    return f"- Voice: {voice}\n- Values: honest work\n"


def user_at_limit() -> str:
    # Each 200-character value produces a 206-character line.  Four of them
    # plus one 176-character line total exactly 1000 characters.
    lines = [f"- {key}: {value * 200}\n" for key, value in (
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
    )]
    lines.append("- E: " + "e" * 170 + "\n")
    text = "".join(lines)
    assert len(text) == USER_LIMIT
    return text


def soul_at_limit() -> str:
    # Seven 200-character values produce 1442 characters; the final line
    # brings the document to exactly 1500 characters.
    lines = [
        f"- {key}: {value * 200}\n"
        for key, value in (
            ("A", "a"),
            ("B", "b"),
            ("C", "c"),
            ("D", "d"),
            ("E", "e"),
            ("F", "f"),
            ("G", "g"),
        )
    ]
    lines.append("- H: " + "h" * 52 + "\n")
    text = "".join(lines)
    assert len(text) == SOUL_LIMIT
    return text


def read_gzip(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8")


def test_limits_are_enforced_on_uncompressed_text() -> None:
    user = IdentityMonitor("user")
    assert user.limit == USER_LIMIT == 1000
    assert user.validate(user_at_limit()).ok

    over_user = user.validate(user_at_limit() + "- F: x\n")
    assert not over_user.ok
    assert any(issue.code == "E_TOO_LONG" for issue in over_user.issues)

    soul = IdentityMonitor("soul")
    assert soul.limit == SOUL_LIMIT == 1500
    assert soul.validate(soul_at_limit()).ok

    over_soul = soul.validate(soul_at_limit() + "- I: x\n")
    assert not over_soul.ok
    assert any(issue.code == "E_TOO_LONG" for issue in over_soul.issues)

    # A document full of multi-byte characters is still measured in code
    # points, not bytes.
    multibyte = "- Nome: " + "界" * 100 + "\n"
    result = user.validate(multibyte)
    assert result.ok
    assert result.chars == len(multibyte)


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        ("Name: Jaden\n", "E_FORMAT"),
        ("- Name Jaden\n", "E_FORMAT"),
        ("- Name:\n", "E_FORMAT"),
        ("- : Jaden\n", "E_KEY"),
        ("- 1Name: Jaden\n", "E_KEY"),
        ("- Name!: Jaden\n", "E_KEY"),
        ("- " + "K" * 41 + ": value\n", "E_KEY"),
        ("- Name: " + "x" * 201 + "\n", "E_VALUE"),
        ("- Name: TODO\n", "E_PLACEHOLDER"),
        ("- Name: N/A\n", "E_PLACEHOLDER"),
        ("- Password: hunter2\n", "E_SECRET"),
        ("- Name: sk-" + "a" * 20 + "\n", "E_SECRET"),
        ("- Name: Jaden\n- name: Jayden\n", "E_DUPLICATE"),
        ("- Name: Ja\x00den\n", "E_CHARS"),
        ("- Name: <soul_edit>\n", "E_SENTINEL"),
    ],
)
def test_strict_format_rejections(candidate: str, code: str) -> None:
    result = IdentityMonitor("user").validate(candidate)
    assert not result.ok
    assert any(issue.code == code for issue in result.issues), result.issues


def test_valid_bullets_and_canonicalization() -> None:
    monitor = IdentityMonitor("user")
    result = monitor.validate(
        "\r\n- Name: Jaden\r\n\r\n- Likes: potatoes\r\n"
    )
    assert result.ok
    assert result.text == "- Name: Jaden\n- Likes: potatoes\n"
    assert result.chars == len(result.text)
    assert result.bullets == (
        ("Name", "Jaden"),
        ("Likes", "potatoes"),
    )


def test_ensure_creates_documents_and_active_backups(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.ensure("user") == ""
    assert store.ensure("soul") == ""

    assert store.document_path("user").read_text(encoding="utf-8") == ""
    assert store.document_path("soul").read_text(encoding="utf-8") == ""
    assert read_gzip(store.active_backup_path("user")) == ""
    assert read_gzip(store.active_backup_path("soul")) == ""

    user_backups = store.list_backups("user")
    assert len(user_backups) == 1
    assert user_backups[0].active
    assert user_backups[0].valid
    assert user_backups[0].chars == 0

    # Re-ensuring must not churn backups.
    assert store.ensure("user") == ""
    assert len(store.list_backups("user")) == 1


def test_apply_writes_full_replacement_and_rotates_backup(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    first = user_document()
    result = store.apply("user", first)
    assert result.status == "accepted"
    assert result.validation.ok
    assert store.document_path("user").read_text(encoding="utf-8") == first
    assert read_gzip(store.active_backup_path("user")) == first

    replacement = "- Name: Jaden\n"
    result = store.apply("user", replacement)
    assert result.status == "accepted"
    assert store.document_path("user").read_text(encoding="utf-8") == replacement
    archived = {
        read_gzip(info.path)
        for info in store.list_backups("user")
        if not info.active
    }
    assert first in archived
    assert "" in archived


def test_retention_keeps_active_plus_two_archives(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.config.retention == DEFAULT_RETENTION == 3
    store.ensure("user")

    documents = [user_document(Age=str(age)) for age in range(24, 29)]
    for document in documents:
        assert store.apply("user", document).status == "accepted"

    backups = store.list_backups("user")
    assert len(backups) == 3
    active = [item for item in backups if item.active]
    archives = [item for item in backups if not item.active]
    assert len(active) == 1
    assert len(archives) == 2
    assert read_gzip(active[0].path) == documents[-1]
    assert {read_gzip(item.path) for item in archives} == {
        documents[-3],
        documents[-2],
    }
    assert all(item.valid for item in backups)


def test_unchanged_edit_is_noop(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    document = user_document()
    assert store.apply("user", document).status == "accepted"
    before = len(store.list_backups("user"))

    result = store.apply("user", document.replace("\n", "\r\n"))
    assert result.status == "unchanged"
    assert len(store.list_backups("user")) == before
    assert store.document_path("user").read_text(encoding="utf-8") == document


def test_rejected_edit_never_changes_files(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    result = store.apply("user", "Name: Jaden\n")
    assert result.status == "rejected"
    assert any(issue.code == "E_FORMAT" for issue in result.validation.issues)
    assert store.document_path("user").read_text(encoding="utf-8") == ""
    assert read_gzip(store.active_backup_path("user")) == ""
    assert len(store.list_backups("user")) == 1


def test_backups_are_lossless_gzip_with_manifest_hashes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    document = "- Nome: João\n- Cidade: São Paulo\n- Emoji: 🚀\n"
    assert store.apply("user", document).status == "accepted"

    active = store.active_backup_path("user")
    assert gzip.decompress(active.read_bytes()).decode("utf-8") == document
    assert read_gzip(active) == document
    info = store.list_backups("user")[0]
    assert info.active
    assert info.chars == len(document)

    manifest = json.loads(
        store.manifest_path("user").read_text(encoding="utf-8")
    )
    assert manifest["codec"] == "gzip"
    assert manifest["active"] == active.name
    entries = {entry["file"]: entry for entry in manifest["entries"]}
    entry = entries[active.name]
    assert entry["chars"] == len(document)
    assert entry["compressed_sha256"] == hashlib.sha256(
        active.read_bytes()
    ).hexdigest()
    assert entry["plaintext_sha256"] == hashlib.sha256(
        document.encode("utf-8")
    ).hexdigest()


def test_corrupt_primary_falls_back_to_active_backup(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    store.apply("user", user_document(Age="24"))
    latest = user_document(Age="25")
    store.apply("user", latest)
    store.document_path("user").write_text("garbage\n", encoding="utf-8")

    loaded = store.load("user")
    assert loaded.source == "backup"
    assert loaded.content == latest
    assert store.document_path("user").read_text(encoding="utf-8") == (
        "garbage\n"
    )


def test_missing_primary_recovers_from_active_backup(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    document = user_document()
    assert store.apply("user", document).status == "accepted"
    store.document_path("user").unlink()

    loaded = store.load("user")
    assert loaded.source == "backup"
    assert loaded.content == document

    assert store.ensure("user") == document
    assert store.document_path("user").read_text(encoding="utf-8") == document


def test_corrupt_primary_refuses_overwrite(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    store.apply("user", user_document())
    store.document_path("user").write_text("garbage\n", encoding="utf-8")

    result = store.apply("user", user_document(Age="30"))
    assert result.status == "rejected"
    assert any(issue.code == "E_CORRUPT" for issue in result.validation.issues)
    assert store.document_path("user").read_text(encoding="utf-8") == (
        "garbage\n"
    )


def test_corrupt_active_backup_is_replaced_on_ensure(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    document = user_document()
    store.apply("user", document)
    store.active_backup_path("user").write_bytes(b"not a gzip stream")

    assert store.ensure("user") == document
    assert read_gzip(store.active_backup_path("user")) == document
    assert store.list_backups("user")[0].valid
    corrupt = list(store.backup_dir("user").glob("*.corrupt.gz"))
    assert len(corrupt) == 1


def test_restore_from_archive(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    first = user_document(Age="24")
    second = user_document(Age="25")
    third = user_document(Age="26")
    for document in (first, second, third):
        assert store.apply("user", document).status == "accepted"

    archive = next(
        item.path
        for item in store.list_backups("user")
        if not item.active and read_gzip(item.path) == first
    )
    result = store.restore("user", archive)
    assert result.status == "accepted"
    assert store.document_path("user").read_text(encoding="utf-8") == first


def test_restore_rejects_manifest_checksum_mismatch(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    original = user_document()
    assert store.apply("user", original).status == "accepted"

    forged = gzip.compress(b"- Name: Evil\n", compresslevel=9)
    store.active_backup_path("user").write_bytes(forged)

    result = store.restore("user", store.active_backup_path("user"))
    assert result.status == "rejected"
    assert any(issue.code == "E_BACKUP" for issue in result.validation.issues)
    assert store.document_path("user").read_text(encoding="utf-8") == original


def test_agents_md_is_never_loaded(tmp_path: Path) -> None:
    root = tmp_path / "identity"
    root.mkdir()
    (root / "AGENTS.md").write_text(
        "- Name: Intruder\n- Secret: nope\n", encoding="utf-8"
    )
    store = make_store(tmp_path)
    assert store.ensure("user") == ""
    assert store.ensure("soul") == ""

    assert store.load("user").source == "primary"
    assert store.load("user").content == ""
    assert store.load("soul").content == ""
    section = store.prompt_section()
    assert "Intruder" not in section
    assert "AGENTS" not in section
    assert store.document_path("user").name == "USER.md"
    assert store.document_path("soul").name == "SOUL.md"


def test_prompt_section_order_sizes_and_protocol(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    store.ensure("soul")
    user = user_document()
    soul = soul_document()
    store.apply("user", user)
    store.apply("soul", soul)

    section = store.prompt_section()
    assert section.startswith("## PERSISTENT USER PROFILE")
    assert f"Size: {len(user)} / {USER_LIMIT} chars" in section
    assert f"Size: {len(soul)} / {SOUL_LIMIT} chars" in section
    assert user.rstrip("\n") in section
    assert soul.rstrip("\n") in section
    assert section.index("<<<USER") < section.index("<<<SOUL")
    assert section.index("<<<SOUL") < section.index(
        "## IDENTITY UPDATE PROTOCOL"
    )
    assert "USER.md is the only place for durable facts" in section
    assert "SOUL.md changes only when the user explicitly asks" in section
    assert "Never put those facts in session state" in section


def test_redaction_removes_known_profile_values(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure("user")
    store.apply("user", user_document())

    text = "I'm Jaden, 24, and I like potatoes."
    redacted = store.redact_for_memory(text)
    for secret in ("Jaden", "24", "potatoes"):
        assert secret not in redacted
    assert "[profile]" in redacted

    assert store.redact_for_memory("Jaden") == "[profile]"


def test_redaction_matches_profile_patterns_without_a_profile(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.ensure("user")

    text = (
        "My name is Jaden and my email is jaden@example.com. "
        "Call me Jay. I'm 24 years old. I live in Berlin."
    )
    redacted = store.redact_for_memory(text)
    for secret in ("Jaden", "jaden@example.com", "Jay", "Berlin"):
        assert secret not in redacted
    assert "24 years old" not in redacted


def test_redaction_strips_edit_blocks_and_withholds_empty_text(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.ensure("user")

    redacted = store.redact_for_memory(
        "Here you go.\n"
        "<user_edit>\n- Name: Jaden\n</user_edit>\n"
        "Done."
    )
    assert "<user_edit>" not in redacted
    assert "Jaden" not in redacted
    assert "Here you go." in redacted
    assert "Done." in redacted

    assert (
        store.redact_for_memory(
            "<user_edit>\n- Name: Jaden\n</user_edit>"
        )
        == PROFILE_WITHHELD
    )
    assert store.redact_for_memory("   \n\t") == PROFILE_WITHHELD


def test_extract_identity_edits_handles_both_blocks() -> None:
    response = (
        "Visible answer.\n"
        "<user_edit>\n- Name: Jaden\n</user_edit>\n"
        "<soul_edit>\n- Voice: dry\n</soul_edit>"
    )
    parsed = extract_identity_edits(response)
    assert parsed.issues == ()
    assert parsed.text == "Visible answer."
    assert parsed.user_edit == "- Name: Jaden"
    assert parsed.soul_edit == "- Voice: dry"


@pytest.mark.parametrize(
    "response",
    [
        "<user_edit>\n- Name: Jaden\n",
        "<user_edit>a</user_edit><user_edit>b</user_edit>",
        "<soul_edit>double</soul_edit><soul_edit>double</soul_edit>",
    ],
)
def test_extract_identity_edits_rejects_malformed_blocks(
    response: str,
) -> None:
    parsed = extract_identity_edits(response)
    assert parsed.user_edit is None
    assert parsed.soul_edit is None
    assert any(issue.code == "E_PROTOCOL" for issue in parsed.issues)
    assert "<user_edit>" not in parsed.text
    assert "<soul_edit>" not in parsed.text


def test_default_identity_is_opt_out(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SORA_IDENTITY", "0")
    assert default_identity() is None

    monkeypatch.setenv("SORA_IDENTITY", "1")
    monkeypatch.setenv("SORA_IDENTITY_ROOT", str(tmp_path / "identity"))
    monkeypatch.setenv("SORA_IDENTITY_RETENTION", "2")
    store = default_identity()
    assert store is not None
    assert store.config.root == tmp_path / "identity"
    assert store.config.retention == 2
