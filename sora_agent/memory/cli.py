"""Command-line entry points for memory operations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from .hook import MemoryHook, default_memory
from .maintenance import optimize_active


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sora-memory")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="show memory configuration and health")
    doctor.add_argument("--json", action="store_true")

    dream = sub.add_parser("dream", help="transfer daily memory and run consolidation")
    dream.add_argument("--dry-run", action="store_true")
    dream.add_argument("--today", action="store_true", help="include today's daily file")

    backup = sub.add_parser("backup", help="create compressed snapshots")
    backup.add_argument("--active", action="store_true")
    backup.add_argument("--all", action="store_true")
    backup.add_argument("--daily", action="append", default=[], metavar="YYYY-MM-DD")

    restore = sub.add_parser("restore", help="restore the active database from a snapshot")
    restore.add_argument("snapshot")
    restore.add_argument("--yes", action="store_true")

    recall = sub.add_parser("recall", help="recall from daily and long-term memory")
    recall.add_argument("query")
    recall.add_argument("--scope", default="auto", choices=["auto", "daily", "longterm"])
    recall.add_argument("--k", type=int, default=8)

    daily = sub.add_parser("daily-search", help="search today's verbatim memory")
    daily.add_argument("query")
    daily.add_argument("--k", type=int, default=8)

    sub.add_parser("download", help="download and verify the CortexDB sidecar")
    sub.add_parser("start", help="start the CortexDB sidecar")
    sub.add_parser("stop", help="stop the CortexDB sidecar")

    optimize = sub.add_parser("optimize", help="offline SQLite maintenance")
    optimize.add_argument("--vacuum", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hook = default_memory(auto_dream=False)
    if hook is None:
        print("memory is disabled (SORA_MEMORY=0)", file=sys.stderr)
        return 2
    try:
        if args.command == "doctor":
            return _doctor(hook, args)
        if args.command == "dream":
            report = hook.dream(dry_run=args.dry_run, include_today=args.today)
            print(report.get("report_path", ""))
            if report.get("errors"):
                print("errors: " + json.dumps(report["errors"]), file=sys.stderr)
            return 0
        if args.command == "backup":
            return _backup(hook, args)
        if args.command == "restore":
            return _restore(hook, args)
        if args.command == "recall":
            print(hook.recall(args.query, scope=args.scope, k=args.k))
            return 0
        if args.command == "daily-search":
            hits = hook.daily.search(args.query, k=args.k, days=7)
            for hit in hits:
                print(
                    f"[{hit.day.isoformat()}#{hit.message_id}] {hit.role}: {hit.content}"
                )
            return 0
        if args.command == "download":
            print(hook.sidecar.download())
            return 0
        if args.command == "start":
            hook.sidecar.ensure_running()
            print(json.dumps(hook.sidecar.doctor(), indent=2, sort_keys=True))
            print("sidecar running; press Ctrl-C to stop", file=sys.stderr)
            try:
                while hook.sidecar.health():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            return 0
        if args.command == "stop":
            hook.sidecar.stop()
            print("stopped")
            return 0
        if args.command == "optimize":
            return _optimize(hook, args)
    except Exception as exc:  # noqa: BLE001
        print("error: " + str(exc), file=sys.stderr)
        return 1
    finally:
        hook.close()
    return 2


def _doctor(hook: MemoryHook, args: argparse.Namespace) -> int:
    payload = hook.doctor()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("root: " + payload["config"]["root"])
        print("active db: " + payload["config"]["db_path"])
        print("backup dir: " + payload["config"]["backup_dir"])
        print("sidecar: " + json.dumps(payload["sidecar"], sort_keys=True))
        print("daily: " + json.dumps(payload["daily"], sort_keys=True))
        print("backup: " + json.dumps(payload["backup"], sort_keys=True))
    return 0


def _backup(hook: MemoryHook, args: argparse.Namespace) -> int:
    created: list[dict[str, Any]] = []
    if args.active or args.all:
        created.append(hook.backup.snapshot_active())
    if args.all or args.daily:
        days = args.daily or [day.isoformat() for day in hook.daily.all_days()]
        for text in days:
            day = date.fromisoformat(text)
            path = hook.daily.path_for(day)
            if path.exists():
                created.append(hook.backup.snapshot_daily(path, day))
    for manifest in created:
        print(manifest["file"])
    return 0


def _restore(hook: MemoryHook, args: argparse.Namespace) -> int:
    snapshot = Path(args.snapshot).expanduser()
    if not args.yes:
        answer = input(f"restore active memory from {snapshot}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted")
            return 1
    result = hook.backup.restore(snapshot, stop=hook.sidecar.stop)
    print(json.dumps(result, indent=2, sort_keys=True))
    try:
        hook.sidecar.ensure_running()
    except Exception as exc:  # noqa: BLE001
        print("restore complete; sidecar did not restart: " + str(exc), file=sys.stderr)
        return 1
    return 0


def _optimize(hook: MemoryHook, args: argparse.Namespace) -> int:
    hook.sidecar.stop()
    if hook.config.db_path.exists():
        hook.backup.snapshot_active()
    result = optimize_active(hook.config, vacuum=args.vacuum)
    print(json.dumps(result, indent=2, sort_keys=True))
    try:
        hook.sidecar.ensure_running()
    except Exception as exc:  # noqa: BLE001
        print("optimized; sidecar did not restart: " + str(exc), file=sys.stderr)
        return 1
    return 0 if result.get("ok") else 1
