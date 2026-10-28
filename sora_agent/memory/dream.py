"""Nightly consolidation of daily memory into active memory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import date, datetime, time as clock_time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .backup import BackupManager
from .client import MemoryBackend
from .config import MemoryConfig
from .daily import DailyStore
from .louvain import louvain_communities

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


class DreamError(RuntimeError):
    pass


class DreamBusy(DreamError):
    pass


class DreamRunner:
    def __init__(
        self,
        config: MemoryConfig,
        daily: DailyStore,
        backend: MemoryBackend,
        backup: Optional[BackupManager] = None,
        wake: Optional[Callable[[Dict[str, Any]], str]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config
        self.daily = daily
        self.backend = backend
        self.backup = backup or BackupManager(config, clock=clock)
        self.wake = wake
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.started = time.monotonic()
        self.report: Dict[str, Any] = {}

    def now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def run(self, dry_run: bool = False, include_today: bool = False) -> Dict[str, Any]:
        config = self.config
        config.ensure_directories()
        now = self.now()
        self.started = time.monotonic()
        run_id = (
            now.strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + str(os.getpid())
            + "-"
            + str(time.time_ns() % 1_000_000)
        )
        report: Dict[str, Any] = {
            "run_id": run_id,
            "started_at": now.isoformat(),
            "finished_at": "",
            "dry_run": dry_run,
            "transferred_days": [],
            "actions": [],
            "conflicts": [],
            "communities": [],
            "inferences": [],
            "errors": [],
            "wake_requested": False,
            "wake_response": "",
            "caps_hit": [],
        }
        self.report = report
        with self._lock():
            if not dry_run and config.db_path.exists():
                try:
                    manifest = self.backup.snapshot_active()
                    report["actions"].append(
                        {
                            "op": "backup",
                            "file": manifest.get("file"),
                            "inverse": "restore from this snapshot",
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - loss guard
                    raise DreamError("active snapshot failed: " + str(exc)) from exc
            self._transfer(report, now, dry_run, include_today)
            try:
                memories = self.backend.list_memories(config.dream_max_memories)
            except Exception as exc:  # noqa: BLE001 - keep daily files
                report["errors"].append("list memories: " + str(exc))
                memories = []
            by_id = {
                str(item.get("id")): item
                for item in memories
                if isinstance(item, dict) and item.get("id")
            }
            cohort, rounds = self._cohort(by_id, now)
            report["cohort"] = sorted(cohort)
            self._pipeline(report, by_id, cohort, rounds, now, dry_run)
            self._mark(report, by_id, cohort, rounds, now, dry_run)
            if not dry_run and hasattr(self.backend, "sweep_expired"):
                try:
                    self.backend.sweep_expired()
                except Exception as exc:  # noqa: BLE001 - best effort
                    report["errors"].append("sweep_expired: " + str(exc))
            report["finished_at"] = self.now().isoformat()
            report["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            if self.wake is not None and report["wake_requested"]:
                try:
                    report["wake_response"] = self.wake(report) or ""
                except Exception as exc:  # noqa: BLE001 - wake is optional
                    report["errors"].append("wake gate: " + str(exc))
            self._write_report(report)
            return report

    def _transfer(
        self,
        report: Dict[str, Any],
        now: datetime,
        dry_run: bool,
        include_today: bool,
    ) -> None:
        pending = self._pending_days(now, include_today)
        next_due = self._next_midnight(now).isoformat()
        for day in pending:
            messages = self.daily.read_day(day)
            if not messages:
                report["transferred_days"].append(
                    {"day": day.isoformat(), "messages": 0, "deleted": False}
                )
                if not dry_run:
                    self.daily.delete_day(day)
                continue
            if not dry_run:
                try:
                    self.backup.snapshot_daily(self.daily.path_for(day), day)
                except Exception as exc:  # noqa: BLE001 - keep the daily file
                    report["errors"].append(
                        "backup {}: {}".format(day.isoformat(), exc)
                    )
                    continue
            saved: List[str] = []
            failed = False
            for message in messages:
                memory_id = "daily:{}:{}".format(day.isoformat(), message["id"])
                payload = {
                    "memory_id": memory_id,
                    "user_id": self.config.user_id,
                    "session_id": str(message.get("session_id") or "default"),
                    "scope": "user",
                    "role": str(message.get("role") or ""),
                    "content": str(message.get("content") or ""),
                    "importance": 0.6 if message.get("role") == "user" else 0.5,
                    "metadata": {
                        "sora_kind": "verbatim",
                        "sora_role": str(message.get("role") or ""),
                        "source": "daily",
                        "source_day": day.isoformat(),
                        "source_message_id": int(message["id"]),
                        "dream_round": 0,
                        "dream_due": next_due,
                    },
                }
                try:
                    if not dry_run:
                        self.backend.save_memory(payload)
                        if self.backend.get_memory(memory_id) is None:
                            raise DreamError("verification failed for " + memory_id)
                    saved.append(memory_id)
                    report["actions"].append(
                        {
                            "op": "save",
                            "memory_id": memory_id,
                            "inverse": "delete " + memory_id,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - keep daily
                    failed = True
                    report["errors"].append(
                        "save {}: {}".format(memory_id, exc)
                    )
                    break
            if failed:
                continue
            if not dry_run:
                self.daily.delete_day(day)
            report["transferred_days"].append(
                {
                    "day": day.isoformat(),
                    "messages": len(saved),
                    "deleted": not dry_run,
                }
            )

    def _cohort(
        self, by_id: Dict[str, Dict[str, Any]], now: datetime
    ) -> Tuple[List[str], Dict[str, int]]:
        cohort: List[str] = []
        rounds: Dict[str, int] = {}
        for memory_id in sorted(by_id):
            record = by_id[memory_id]
            metadata = _metadata(record)
            try:
                current_round = int(metadata.get("dream_round") or 0)
            except (TypeError, ValueError):
                current_round = 0
            due = metadata.get("dream_due")
            if current_round <= 0:
                cohort.append(memory_id)
                rounds[memory_id] = 1
            elif current_round < self.config.dream_rounds:
                eligible = True
                if due:
                    try:
                        eligible = _parse_iso(str(due)) <= now
                    except (TypeError, ValueError):
                        eligible = True
                if eligible:
                    cohort.append(memory_id)
                    rounds[memory_id] = current_round + 1
        return cohort[: self.config.dream_max_memories], rounds

    def _pipeline(
        self,
        report: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
        cohort: List[str],
        rounds: Dict[str, int],
        now: datetime,
        dry_run: bool,
    ) -> None:
        records = [by_id[memory_id] for memory_id in cohort if memory_id in by_id]
        if not records:
            return
        if self._over_budget(report):
            return
        self._dedupe(report, by_id, records, rounds, now, dry_run)
        self._conflicts(report, by_id, records, rounds, now, dry_run)
        if self._over_budget(report):
            return
        graph = self._graph(records)
        communities = louvain_communities(graph)
        self._communities(report, records, communities, dry_run)
        self._inferences(report, by_id, records, graph, communities, rounds, now, dry_run)
        if self.config.dream_rules:
            self._rules(report, dry_run)
        self._decay(report, records, now, dry_run)
        self._expire(report, records, now, dry_run)

    def _dedupe(
        self,
        report: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
        records: List[Dict[str, Any]],
        rounds: Dict[str, int],
        now: datetime,
        dry_run: bool,
    ) -> None:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            metadata = _metadata(record)
            if _kind(metadata) in {"merged", "conflict", "inferred_link"}:
                continue
            normalized = _normalize(str(record.get("content") or ""))
            if len(normalized) < 40:
                continue
            groups.setdefault(normalized, []).append(record)
        created = 0
        for normalized in sorted(groups):
            group = groups[normalized]
            if len(group) < 2 or created >= self.config.dream_max_merges:
                continue
            ids = sorted(str(item.get("id")) for item in group)
            merged_id = "dream:merge:" + _digest("|".join(ids))
            if merged_id in by_id:
                continue
            target_round = max(rounds.get(memory_id, 1) for memory_id in ids)
            canonical = max(
                group,
                key=lambda item: (
                    str(item.get("created_at") or ""),
                    len(str(item.get("content") or "")),
                ),
            )
            metadata = {
                "sora_kind": "merged",
                "sources": ids,
                "dream_round": target_round,
                "dream_due": self._due_for(target_round, now),
            }
            payload = {
                "memory_id": merged_id,
                "user_id": self.config.user_id,
                "scope": "user",
                "content": str(canonical.get("content") or ""),
                "importance": max(_importance(item) for item in group),
                "metadata": metadata,
                "supersedes": [memory_id for memory_id in ids if memory_id != str(canonical.get("id"))],
            }
            report["actions"].append(
                {
                    "op": "merge",
                    "memory_id": merged_id,
                    "sources": ids,
                    "inverse": "delete merged memory; superseded sources remain stored",
                }
            )
            created += 1
            if dry_run:
                continue
            try:
                self.backend.save_memory(payload)
                by_id[merged_id] = {
                    "id": merged_id,
                    "content": payload["content"],
                    "metadata": metadata,
                    "importance": payload["importance"],
                }
            except Exception as exc:  # noqa: BLE001 - non-fatal phase
                report["errors"].append("merge {}: {}".format(merged_id, exc))

    def _conflicts(
        self,
        report: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
        records: List[Dict[str, Any]],
        rounds: Dict[str, int],
        now: datetime,
        dry_run: bool,
    ) -> None:
        candidates = [
            record
            for record in records
            if _kind(_metadata(record)) not in {"conflict", "inferred_link"}
        ]
        tokens = {str(item.get("id")): _tokens(item) for item in candidates}
        created = 0
        for left_index, left in enumerate(candidates):
            if created >= self.config.dream_max_conflicts:
                break
            left_id = str(left.get("id"))
            left_text = str(left.get("content") or "")
            for right in candidates[left_index + 1:]:
                if created >= self.config.dream_max_conflicts:
                    break
                right_id = str(right.get("id"))
                right_text = str(right.get("content") or "")
                if _jaccard(tokens[left_id], tokens[right_id]) < 0.7:
                    continue
                if _numbers(left_text) == _numbers(right_text):
                    continue
                conflict_id = "dream:conflict:" + _digest(
                    "|".join(sorted([left_id, right_id]))
                )
                if conflict_id in by_id:
                    continue
                target_round = max(
                    rounds.get(left_id, 1), rounds.get(right_id, 1)
                )
                content = "Conflicting records:\n- [{}] {}\n- [{}] {}".format(
                    left_id, _brief(left_text), right_id, _brief(right_text)
                )
                payload = {
                    "memory_id": conflict_id,
                    "user_id": self.config.user_id,
                    "scope": "user",
                    "content": content,
                    "importance": 0.8,
                    "metadata": {
                        "sora_kind": "conflict",
                        "sources": [left_id, right_id],
                        "dream_round": target_round,
                        "dream_due": self._due_for(target_round, now),
                    },
                }
                report["conflicts"].append(
                    {
                        "memory_id": conflict_id,
                        "sources": [left_id, right_id],
                    }
                )
                report["wake_requested"] = True
                report["actions"].append(
                    {
                        "op": "conflict",
                        "memory_id": conflict_id,
                        "sources": [left_id, right_id],
                        "inverse": "delete " + conflict_id,
                    }
                )
                created += 1
                if dry_run:
                    continue
                try:
                    self.backend.save_memory(payload)
                    by_id[conflict_id] = {
                        "id": conflict_id,
                        "content": content,
                        "metadata": payload["metadata"],
                        "importance": 0.8,
                    }
                except Exception as exc:  # noqa: BLE001 - non-fatal phase
                    report["errors"].append(
                        "conflict {}: {}".format(conflict_id, exc)
                    )

    def _graph(self, records: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        token_sets = {str(item.get("id")): _tokens(item) for item in records}
        ids = sorted(token_sets)
        edges: List[Tuple[str, str, float]] = []
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                shared = token_sets[left] & token_sets[right]
                if len(shared) < 2:
                    continue
                weight = _jaccard(token_sets[left], token_sets[right])
                if weight >= self.config.dream_edge_jaccard:
                    edges.append((left, right, weight))
        edges.sort()
        return edges

    def _communities(
        self,
        report: Dict[str, Any],
        records: List[Dict[str, Any]],
        communities: Dict[str, int],
        dry_run: bool,
    ) -> None:
        groups: Dict[int, List[str]] = {}
        for memory_id, label in communities.items():
            groups.setdefault(int(label), []).append(memory_id)
        report["communities"] = [
            sorted(members) for _, members in sorted(groups.items())
        ]
        updated = 0
        for record in records:
            memory_id = str(record.get("id"))
            label = communities.get(memory_id)
            if label is None or updated >= self.config.dream_max_updates:
                continue
            metadata = _metadata(record)
            if metadata.get("dream_community") == label:
                continue
            report["actions"].append(
                {
                    "op": "community",
                    "memory_id": memory_id,
                    "community": label,
                    "inverse": "remove dream_community",
                }
            )
            updated += 1
            if dry_run:
                continue
            try:
                self.backend.update_memory(
                    memory_id, metadata={"dream_community": label}
                )
            except Exception as exc:  # noqa: BLE001 - non-fatal phase
                report["errors"].append(
                    "community {}: {}".format(memory_id, exc)
                )

    def _inferences(
        self,
        report: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
        records: List[Dict[str, Any]],
        graph: List[Tuple[str, str, float]],
        communities: Dict[str, int],
        rounds: Dict[str, int],
        now: datetime,
        dry_run: bool,
    ) -> None:
        adjacency: Dict[str, Set[str]] = {}
        for left, right, _ in graph:
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        created = 0
        for middle in sorted(adjacency):
            if created >= self.config.dream_max_inferences:
                break
            neighbours = sorted(adjacency[middle])
            for index, left in enumerate(neighbours):
                for right in neighbours[index + 1:]:
                    if right in adjacency.get(left, set()):
                        continue
                    if communities.get(left) != communities.get(right):
                        continue
                    link_id = "dream:link:" + _digest(
                        "|".join(sorted([left, right]))
                    )
                    if link_id in by_id:
                        continue
                    target_round = max(
                        rounds.get(left, 1), rounds.get(right, 1)
                    )
                    left_text = _brief(str(by_id.get(left, {}).get("content") or ""))
                    right_text = _brief(str(by_id.get(right, {}).get("content") or ""))
                    content = (
                        "Inferred two-hop relation through {}: [{}] {} <-> [{}] {}"
                    ).format(middle, left, left_text, right, right_text)
                    payload = {
                        "memory_id": link_id,
                        "user_id": self.config.user_id,
                        "scope": "user",
                        "content": content,
                        "importance": 0.4,
                        "metadata": {
                            "sora_kind": "inferred_link",
                            "sources": [left, middle, right],
                            "dream_round": target_round,
                            "dream_due": self._due_for(target_round, now),
                        },
                    }
                    report["inferences"].append(
                        {"memory_id": link_id, "sources": [left, middle, right]}
                    )
                    report["actions"].append(
                        {
                            "op": "infer",
                            "memory_id": link_id,
                            "sources": [left, middle, right],
                            "inverse": "delete " + link_id,
                        }
                    )
                    created += 1
                    if dry_run:
                        continue
                    try:
                        self.backend.save_memory(payload)
                        by_id[link_id] = {
                            "id": link_id,
                            "content": content,
                            "metadata": payload["metadata"],
                            "importance": 0.4,
                        }
                    except Exception as exc:  # noqa: BLE001 - non-fatal phase
                        report["errors"].append(
                            "inference {}: {}".format(link_id, exc)
                        )

    def _rules(self, report: Dict[str, Any], dry_run: bool) -> None:
        try:
            preview = self.backend.rules_apply(
                rules=list(self.config.dream_rules),
                dry_run=True,
                max_derived=1000,
            )
            report["actions"].append(
                {"op": "rules_preview", "result": preview}
            )
            if not dry_run:
                applied = self.backend.rules_apply(
                    rules=list(self.config.dream_rules),
                    dry_run=False,
                    max_derived=1000,
                )
                report["actions"].append(
                    {
                        "op": "rules_apply",
                        "result": applied,
                        "inverse": "inference_explain",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - optional optimization
            report["errors"].append("rules_apply: " + str(exc))

    def _decay(
        self,
        report: Dict[str, Any],
        records: List[Dict[str, Any]],
        now: datetime,
        dry_run: bool,
    ) -> None:
        updates = 0
        for record in records:
            if updates >= self.config.dream_max_updates:
                report["caps_hit"].append("dream_max_updates")
                break
            memory_id = str(record.get("id"))
            previous = _importance(record)
            candidate = max(
                self.config.dream_min_importance,
                round(previous * self.config.dream_decay, 4),
            )
            if abs(candidate - previous) < 0.01:
                continue
            report["actions"].append(
                {
                    "op": "decay",
                    "memory_id": memory_id,
                    "from": previous,
                    "to": candidate,
                    "inverse": previous,
                }
            )
            updates += 1
            if dry_run:
                continue
            try:
                self.backend.update_memory(
                    memory_id,
                    importance=candidate,
                    metadata={"last_decay": now.isoformat()},
                )
            except Exception as exc:  # noqa: BLE001 - non-fatal phase
                report["errors"].append("decay {}: {}".format(memory_id, exc))

    def _expire(
        self,
        report: Dict[str, Any],
        records: List[Dict[str, Any]],
        now: datetime,
        dry_run: bool,
    ) -> None:
        removed = 0
        for record in records:
            if removed >= self.config.dream_max_updates:
                break
            memory_id = str(record.get("id"))
            metadata = _metadata(record)
            expires = record.get("expires_at") or metadata.get("sora_expires_at") or metadata.get("expires_at")
            if not expires:
                continue
            try:
                moment = _parse_iso(str(expires))
            except ValueError:
                continue
            if moment > now:
                continue
            report["actions"].append(
                {
                    "op": "expire",
                    "memory_id": memory_id,
                    "inverse": "restore saved content",
                }
            )
            removed += 1
            if dry_run:
                continue
            try:
                self.backend.delete_memory(memory_id)
            except Exception as exc:  # noqa: BLE001 - non-fatal phase
                report["errors"].append("expire {}: {}".format(memory_id, exc))

    def _mark(
        self,
        report: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
        cohort: List[str],
        rounds: Dict[str, int],
        now: datetime,
        dry_run: bool,
    ) -> None:
        updates = 0
        for memory_id in cohort:
            if updates >= self.config.dream_max_updates:
                report["caps_hit"].append("dream_max_updates")
                break
            target_round = rounds.get(memory_id, 1)
            completed = min(target_round, max(1, self.config.dream_rounds))
            next_due = (
                self._next_midnight(now).isoformat()
                if completed < self.config.dream_rounds
                else None
            )
            report["actions"].append(
                {
                    "op": "mark",
                    "memory_id": memory_id,
                    "dream_round": completed,
                    "dream_due": next_due,
                    "inverse": _metadata(by_id.get(memory_id, {})),
                }
            )
            updates += 1
            if dry_run:
                continue
            try:
                self.backend.update_memory(
                    memory_id,
                    metadata={
                        "dream_round": completed,
                        "dream_due": next_due,
                        "dream_last_run": now.isoformat(),
                        "dream_run_id": report["run_id"],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - keep daily already transferred
                report["errors"].append("mark {}: {}".format(memory_id, exc))

    def _pending_days(self, now: datetime, include_today: bool) -> List[date]:
        pending = self.daily.pending_days(now)
        if include_today and self.daily.day_for(now) not in pending:
            pending.append(self.daily.day_for(now))
        return sorted(pending)

    def _next_midnight(self, now: datetime) -> datetime:
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(
            tomorrow,
            clock_time(hour=max(0, min(23, self.config.dream_hour_utc))),
            tzinfo=timezone.utc,
        )

    def _due_for(self, target_round: int, now: datetime) -> Optional[str]:
        if target_round < self.config.dream_rounds:
            return self._next_midnight(now).isoformat()
        return None

    def _over_budget(self, report: Dict[str, Any]) -> bool:
        if time.monotonic() - self.started >= self.config.dream_budget_seconds:
            report["caps_hit"].append("time_budget")
            return True
        return False

    def _write_report(self, report: Dict[str, Any]) -> None:
        self.config.ensure_directories()
        path = self.config.report_dir / (report["run_id"] + ".json")
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        os.replace(str(temp), str(path))
        report["report_path"] = str(path)

    @contextmanager
    def _lock(self):
        self.config.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.config.lock_path, "a+")
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise DreamBusy("another dream is running") from exc
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise DreamBusy("another dream is running") from exc
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()


def _metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _kind(metadata: Dict[str, Any]) -> str:
    value = metadata.get("sora_kind") or metadata.get("kind") or ""
    return str(value)


def _importance(record: Dict[str, Any]) -> float:
    try:
        return float(record.get("importance") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _tokens(record: Dict[str, Any]) -> Set[str]:
    return set(_TOKEN_RE.findall(str(record.get("content") or "").lower()))


def _numbers(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(text))


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _brief(text: str, limit: int = 180) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
