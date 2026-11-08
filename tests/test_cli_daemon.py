"""Tests for the background gateway lifecycle."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

import pytest
from typer.testing import CliRunner

from sora_agent import __version__
from sora_agent.cli import daemon
from sora_agent.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SORA_GATEWAY_PID", str(tmp_path / "gateway.pid"))
    monkeypatch.setenv("SORA_GATEWAY_LOG", str(tmp_path / "gateway.log"))


class _RunningProcess:
    pid = 4321

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("gateway", timeout or 0)


class _ExitedProcess:
    pid = 4322

    def wait(self, timeout=None):
        return 1


def test_pid_record_round_trip(tmp_path):
    daemon._write_process(4321)

    record = daemon._read_process()

    assert record is not None
    assert record.pid == 4321
    assert record.version == __version__
    assert record.started_at
    assert daemon.pid_path().stat().st_mode & 0o777 == 0o600

    daemon._remove_process()
    assert daemon._read_process() is None


def test_pid_alive_detects_self_and_missing_process():
    assert daemon._pid_alive(os.getpid()) is True
    assert daemon._pid_alive(2**31 - 1) is False


def test_start_refuses_to_spawn_a_second_gateway(monkeypatch):
    daemon._write_process(4321)
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(daemon, "_looks_like_gateway", lambda pid: True)
    monkeypatch.setattr(
        daemon.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("gateway was spawned twice"),
    )

    assert daemon.start_gateway() == 0


def test_start_lock_prevents_concurrent_starts():
    with daemon._start_lock() as first:
        assert first is True
        with daemon._start_lock() as second:
            assert second is False

    with daemon._start_lock() as third:
        assert third is True


def test_start_spawns_a_detached_gateway(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _RunningProcess()

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    assert daemon.start_gateway() == 0

    command = captured["command"]
    kwargs = captured["kwargs"]
    assert command[-2:] == ["-m", "sora_agent.channels"]
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    if os.name != "nt":
        assert kwargs["start_new_session"] is True
    record = daemon._read_process()
    assert record is not None
    assert record.pid == 4321
    assert daemon.log_path().stat().st_mode & 0o777 == 0o600
    assert "sora gateway start" in daemon.log_path().read_text()


def test_start_reports_an_immediate_exit(monkeypatch):
    monkeypatch.setattr(
        daemon.subprocess, "Popen", lambda *args, **kwargs: _ExitedProcess()
    )

    assert daemon.start_gateway() == 1
    assert daemon._read_process() is None


def test_stop_terminates_the_gateway_and_removes_the_pid_file(monkeypatch):
    daemon._write_process(4321)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(daemon, "_looks_like_gateway", lambda pid: True)
    monkeypatch.setattr(daemon, "_wait_for_exit", lambda pid, timeout: True)
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    assert daemon.stop_gateway() == 0
    assert signals == [(4321, signal.SIGTERM)]
    assert daemon._read_process() is None


def test_stop_refuses_an_unrelated_process(monkeypatch):
    daemon._write_process(4321)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(daemon, "_looks_like_gateway", lambda pid: False)
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    assert daemon.stop_gateway() == 2
    assert signals == []
    assert daemon._read_process() is not None


def test_stop_removes_a_stale_pid_file(monkeypatch):
    daemon._write_process(4321)
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: False)

    assert daemon.stop_gateway() == 0
    assert daemon._read_process() is None


def test_status_reports_running_and_stopped(monkeypatch):
    assert daemon.status_gateway() == 1

    daemon._write_process(4321)
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(daemon, "_looks_like_gateway", lambda pid: True)

    assert daemon.status_gateway() == 0

    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: False)
    assert daemon.status_gateway() == 1


def test_tail_and_show_logs(tmp_path, capsys):
    path = tmp_path / "gateway.log"
    path.write_text("\n".join(f"line {index}" for index in range(100)) + "\n")

    assert daemon._tail(path, 3) == ["line 97", "line 98", "line 99"]

    assert daemon.show_logs(2) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "line 98",
        "line 99",
    ]


def test_gateway_status_is_exposed_by_the_cli():
    result = runner.invoke(app, ["gateway", "status"])

    assert result.exit_code == 1
    assert "not running" in result.output


def test_gateway_logs_are_exposed_by_the_cli():
    daemon.log_path().write_text("first\nsecond\n")

    result = runner.invoke(app, ["gateway", "logs", "-n", "1"])

    assert result.exit_code == 0
    assert "second" in result.output
