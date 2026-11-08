"""Tests for the optional, calibrated trajectory judge."""

from __future__ import annotations

import json

import pytest

from sora_agent.eval import Trajectory
from sora_agent.judge import (
    Calibration,
    JudgeConfig,
    JudgeGrader,
    calibrate_judge,
    calibration_metrics,
    load_calibration,
    save_calibration,
)
from sora_agent.providers import ModelTurn
from sora_agent.runner import RunMetrics

GOOD_JSON = json.dumps(
    {
        "passed": True,
        "dimensions": {
            "groundedness": 1.0,
            "task_completion": 0.8,
            "tool_selection": 0.9,
            "error_recovery": 0.7,
            "unnecessary_loops": 0.9,
            "safety": 1.0,
        },
        "notes": ["looks fine"],
    }
)


class JsonProvider:
    def __init__(self, text: str = GOOD_JSON) -> None:
        self.text = text
        self.calls = 0

    def complete_turn(self, messages, tools) -> ModelTurn:
        self.calls += 1
        return ModelTurn(text=self.text)


def _calibration(tmp_path, **overrides) -> Calibration:
    values = {
        "agreement": 0.85,
        "kappa": 0.70,
        "recall_bad": 0.90,
        "sample_size": 100,
    }
    values.update(overrides)
    return Calibration(**values)


def _config(tmp_path, **overrides) -> JudgeConfig:
    path = tmp_path / "calibration.json"
    save_calibration(path, _calibration(tmp_path))
    values = {
        "sample_rate": 1.0,
        "calibration_path": path,
        "agent_family": "openai",
        "model_family": "anthropic",
    }
    values.update(overrides)
    return JudgeConfig(**values)


def _trajectory(run_id: str = "run-1") -> Trajectory:
    return Trajectory(
        task="look up an order",
        metrics=RunMetrics(),
        final_answer="done",
        stop_reason="completed",
        run_id=run_id,
    )


def test_judging_is_off_by_default():
    provider = JsonProvider()
    grader = JudgeGrader(provider, JudgeConfig())

    report = grader(_trajectory())

    assert report.skipped is True
    assert provider.calls == 0


def test_sampling_is_deterministic(tmp_path):
    provider = JsonProvider()
    grader = JudgeGrader(provider, _config(tmp_path, sample_rate=0.5))

    first = grader.sampled("stable-run-id")
    second = grader.sampled("stable-run-id")

    assert first is second


def test_judge_refuses_without_calibration():
    provider = JsonProvider()
    config = JudgeConfig(sample_rate=1.0, agent_family="a", model_family="b")
    report = JudgeGrader(provider, config)(_trajectory())

    assert report.error
    assert "not calibrated" in report.error
    assert provider.calls == 0


def test_judge_refuses_below_threshold(tmp_path):
    path = tmp_path / "weak.json"
    save_calibration(
        path,
        _calibration(tmp_path, agreement=0.70, kappa=0.2, recall_bad=0.4),
    )
    config = JudgeConfig(
        sample_rate=1.0,
        calibration_path=path,
        agent_family="a",
        model_family="b",
    )
    report = JudgeGrader(JsonProvider(), config)(_trajectory())

    assert "below threshold" in report.error


def test_judge_refuses_the_same_model_family(tmp_path):
    config = _config(tmp_path, agent_family="openai", model_family="openai")
    report = JudgeGrader(JsonProvider(), config)(_trajectory())

    assert "different model family" in report.error


def test_judge_scores_dimensions(tmp_path):
    report = JudgeGrader(JsonProvider(), _config(tmp_path))(_trajectory())

    assert report.error == ""
    assert report.passed is True
    assert set(report.dimensions) == {
        "groundedness",
        "task_completion",
        "tool_selection",
        "error_recovery",
        "unnecessary_loops",
        "safety",
    }
    assert report.score == pytest.approx(5.3 / 6)
    assert report.checks == ["looks fine"]


def test_malformed_judge_output_becomes_an_error(tmp_path):
    report = JudgeGrader(
        JsonProvider("not json at all"), _config(tmp_path)
    )(_trajectory())

    assert report.error
    assert "JSON" in report.error


def test_safety_is_advisory_only(tmp_path):
    payload = json.loads(GOOD_JSON)
    payload["dimensions"]["safety"] = 0.0
    payload.pop("passed")
    report = JudgeGrader(
        JsonProvider(json.dumps(payload)), _config(tmp_path)
    )(_trajectory())

    assert report.dimensions["safety"] == 0.0
    assert report.passed is True


def test_calibration_metrics():
    calibration = calibration_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert calibration.sample_size == 4
    assert calibration.agreement == 0.5
    assert calibration.recall_bad == 0.5
    assert calibration.kappa == 0.0


def test_calibration_round_trip(tmp_path):
    path = tmp_path / "nested" / "calibration.json"
    original = Calibration(0.8, 0.6, 0.9, 50, "2026-11-03T00:00:00Z")

    save_calibration(path, original)
    loaded = load_calibration(path)

    assert loaded == original


def test_calibration_rejects_non_finite_values(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        Calibration(float("nan"), 0.6, 0.9, 10)
    with pytest.raises(ValueError, match="finite"):
        Calibration(0.8, float("inf"), 0.9, 10)

    path = tmp_path / "nan.json"
    path.write_text(
        json.dumps(
            {
                "agreement": float("nan"),
                "kappa": float("inf"),
                "recall_bad": 0.9,
                "sample_size": 10,
            }
        ),
        encoding="utf-8",
    )
    assert load_calibration(path) is None


def test_calibration_rejects_out_of_range_values(tmp_path):
    with pytest.raises(ValueError):
        Calibration(1.5, 0.6, 0.9, 10)
    with pytest.raises(ValueError):
        Calibration(0.8, -2.0, 0.9, 10)
    with pytest.raises(ValueError):
        Calibration(0.8, 0.6, 1.5, 10)
    with pytest.raises(ValueError):
        Calibration(0.8, 0.6, 0.9, -1)


def test_non_finite_dimensions_are_ignored(tmp_path):
    payload = (
        '{"passed": true, "dimensions": {'
        '"groundedness": NaN, "task_completion": NaN, '
        '"tool_selection": NaN, "error_recovery": NaN, '
        '"unnecessary_loops": NaN, "safety": NaN}}'
    )

    report = JudgeGrader(JsonProvider(payload), _config(tmp_path))(
        _trajectory()
    )

    assert report.error
    assert "no dimension scores" in report.error


def test_finite_dimensions_survive_a_mixed_response(tmp_path):
    payload = json.loads(GOOD_JSON)
    payload["dimensions"]["groundedness"] = float("nan")
    payload["dimensions"]["task_completion"] = 0.2

    report = JudgeGrader(
        JsonProvider(json.dumps(payload)), _config(tmp_path)
    )(_trajectory())

    assert "groundedness" not in report.dimensions
    assert report.dimensions["task_completion"] == 0.2


def test_calibrate_judge_skips_errored_reports(tmp_path):
    def grader(trajectory):
        from sora_agent.eval import EvalReport

        if trajectory.run_id == "bad":
            return EvalReport(False, 0.0, error="boom")
        return EvalReport(trajectory.run_id != "fail", 1.0)

    calibration = calibrate_judge(
        grader,
        [
            (_trajectory("pass"), False),
            (_trajectory("fail"), True),
            (_trajectory("bad"), True),
        ],
    )

    assert calibration.sample_size == 2
    assert calibration.agreement == 1.0
