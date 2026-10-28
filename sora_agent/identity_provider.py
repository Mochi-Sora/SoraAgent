"""Provider adapter for the persistent USER.md/SOUL.md identity system.

The wrapper inserts the identity section into the system prompt and removes
``<user_edit>`` / ``<soul_edit>`` blocks from the visible answer.  Identity is
always inserted before the existing session-state section so the final order is:

    base system prompt -> USER -> SOUL -> state -> user message
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from .identity import (
    IDENTITY_HEADING,
    SOUL_LIMIT,
    STATE_HEADING,
    USER_LIMIT,
    ExtractedIdentityEdits,
    IdentityIssue,
    IdentityStore,
    extract_identity_edits,
)
from .providers import Message

log = logging.getLogger(__name__)


class IdentityProvider:
    """Wrap a provider with persistent identity injection and edit handling."""

    __sora_identity__ = True

    def __init__(
        self,
        inner: Any,
        identity: IdentityStore,
        max_repairs: int | None = None,
    ) -> None:
        self.inner = inner
        self.identity = identity
        self.max_repairs = (
            identity.config.max_repairs
            if max_repairs is None
            else max(0, min(int(max_repairs), 5))
        )

    def complete(self, messages: Sequence[Any], tools: Sequence[Any]) -> str:
        try:
            section = self.identity.prompt_section()
        except Exception as exc:  # noqa: BLE001 - identity must fail open
            log.warning("identity prompt section failed: %s", exc)
            section = ""
        augmented = self._with_identity(messages, section) if section else list(messages)
        raw = self.inner.complete(augmented, tools)
        return self._settle(raw, augmented, tools)

    # ------------------------------------------------------------------
    # Prompt insertion
    # ------------------------------------------------------------------
    @staticmethod
    def _with_identity(messages: Sequence[Any], section: str) -> list[Any]:
        augmented = list(messages)

        # Replace an existing identity section in place rather than duplicating it.
        for index, message in enumerate(augmented):
            if (
                getattr(message, "role", "") == "system"
                and str(getattr(message, "content", "")).startswith(
                    IDENTITY_HEADING
                )
            ):
                augmented[index] = Message("system", section)
                return augmented

        insert_at = 0
        for index, message in enumerate(augmented):
            if getattr(message, "role", "") != "system":
                break
            content = str(getattr(message, "content", ""))
            if content.startswith(STATE_HEADING):
                insert_at = index
                break
            insert_at = index + 1

        augmented.insert(insert_at, Message("system", section))
        return augmented

    # ------------------------------------------------------------------
    # Edit handling
    # ------------------------------------------------------------------
    def _try_apply(
        self,
        parsed: ExtractedIdentityEdits,
        accepted: set[str],
    ) -> tuple[set[str], dict[str, tuple[IdentityIssue, ...]], list[IdentityIssue]]:
        issues = list(parsed.issues)
        rejected: dict[str, tuple[IdentityIssue, ...]] = {}
        if issues:
            return accepted, rejected, issues

        for document, edit in (
            ("user", parsed.user_edit),
            ("soul", parsed.soul_edit),
        ):
            if edit is None or document in accepted:
                continue
            try:
                applied = self.identity.apply(document, edit)
            except Exception as exc:  # noqa: BLE001 - provider must fail open
                rejected[document] = (
                    IdentityIssue(
                        "E_IO",
                        f"Identity storage failed: {exc}",
                        document,
                    ),
                )
                continue
            if applied.status in {"accepted", "unchanged"}:
                accepted.add(document)
            else:
                rejected[document] = applied.validation.issues
        return accepted, rejected, issues

    def _repair_prompt(
        self,
        failed: set[str],
        issues: Iterable[IdentityIssue],
        rejected: dict[str, tuple[IdentityIssue, ...]],
        attempt: int,
    ) -> str:
        issue_lines: list[str] = []
        for issue in issues:
            document = issue.document or "identity"
            issue_lines.append(
                f"- [{document}] [{issue.code}] {issue.message}"
            )
        for document, document_issues in rejected.items():
            for issue in document_issues:
                issue_lines.append(
                    f"- [{document}] [{issue.code}] {issue.message}"
                )
        if not issue_lines:
            issue_lines.append(
                "- [identity] [E_PROTOCOL] No valid identity edit was provided."
            )

        blocks: list[str] = []
        for document in ("user", "soul"):
            if document not in failed:
                continue
            title = "USER" if document == "user" else "SOUL"
            loaded = self.identity.load(document)
            current = loaded.content or "(empty)"
            blocks.append(
                f"Current {title}.md (authoritative):\n"
                f"--- {title} START ---\n{current.rstrip()}\n--- {title} END ---"
            )

        failed_text = ", ".join(
            "USER.md" if item == "user" else "SOUL.md"
            for item in sorted(failed)
        )
        return (
            "The framework's identity monitor rejected the identity edit. "
            "Nothing was written for the rejected document(s); the previous "
            "content is unchanged.\n\n"
            f"Repair attempt {attempt} of {self.max_repairs}.\n\n"
            "Rejected issues:\n" + "\n".join(issue_lines) + "\n\n"
            + "\n\n".join(blocks)
            + "\n\nReturn only one corrected block for each failed document: "
            f"{failed_text}. Use <user_edit>...</user_edit> and/or "
            "<soul_edit>...</soul_edit>. The block must contain the complete "
            "replacement, one fact per line as '- Key: Value', with no "
            "headings or prose. "
            f"USER.md must be at most {USER_LIMIT} characters and SOUL.md at most "
            f"{SOUL_LIMIT} characters. If you cannot produce a valid replacement, "
            "omit the block."
        )

    def _settle(
        self,
        raw: str,
        base_messages: Sequence[Any],
        tools: Sequence[Any],
    ) -> str:
        parsed = extract_identity_edits(raw)
        visible = parsed.text
        accepted: set[str] = set()
        accepted, rejected, issues = self._try_apply(parsed, accepted)
        if not issues and not rejected:
            return visible

        raw_for_repair = raw
        for attempt in range(1, self.max_repairs + 1):
            failed = set(rejected)
            for issue in issues:
                if issue.document in {"user", "soul"}:
                    failed.add(issue.document)
            if not failed:
                failed = {"user", "soul"}

            repair = self._repair_prompt(
                failed, issues, rejected, attempt
            )
            repair_messages = list(base_messages) + [
                Message("assistant", raw_for_repair),
                Message("user", repair),
            ]
            raw_for_repair = self.inner.complete(repair_messages, tools)
            parsed = extract_identity_edits(raw_for_repair)
            accepted, rejected, issues = self._try_apply(parsed, accepted)
            if not issues and not rejected:
                # The repair response may contain a new visible answer, but the
                # original answer is the one the user sees.
                return visible

        log.warning(
            "identity edit dropped after %s repair attempt(s)", self.max_repairs
        )
        return visible
