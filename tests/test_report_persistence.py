"""Tests for GuardReport.save() and GuardReport.load() (Feature 3 — Report Persistence)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from longguard.core.breaker import BreakerState
from longguard.core.reporter import DetectionEvent, GuardReport
from longguard.core.step import AgentStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_report() -> GuardReport:
    """Build a realistic GuardReport with detections and timeline."""
    report = GuardReport(model="gpt-4o")
    for i in range(1, 4):
        step = AgentStep(step_number=i, thought=f"Thought {i}", tokens_used=100 * i)
        report.record_step(step, total_tokens=100 * i, total_steps=i, cost_delta=0.001 * i)

    report.detections.append(
        DetectionEvent(step_number=2, pattern="tool_repeat", confidence=0.9, evidence={"count": 3})
    )
    report.record_reflection()
    report.finalize(BreakerState.OPEN, kill_reason="reflection_failed: tool_repeat")
    return report


# ===========================================================================
# DetectionEvent.from_dict round-trip
# ===========================================================================


class TestDetectionEventFromDict:
    def test_round_trip(self) -> None:
        det = DetectionEvent(
            step_number=5, pattern="dead_end_drift", confidence=0.75, evidence={"steps": 5}
        )
        restored = DetectionEvent.from_dict(det.to_dict())
        assert restored.step_number == det.step_number
        assert restored.pattern == det.pattern
        assert restored.confidence == det.confidence
        assert restored.evidence == det.evidence

    def test_missing_evidence_defaults_to_empty(self) -> None:
        data = {"step_number": 1, "pattern": "tool_repeat", "confidence": 0.8}
        det = DetectionEvent.from_dict(data)
        assert det.evidence == {}


# ===========================================================================
# GuardReport.to_dict / from_dict round-trip
# ===========================================================================


class TestGuardReportFromDict:
    def test_to_dict_contains_all_fields(self) -> None:
        report = _build_report()
        d = report.to_dict()

        assert d["total_steps"] == 3
        assert d["total_tokens"] == 300
        assert d["estimated_cost_usd"] == pytest.approx(0.006)
        assert d["model"] == "gpt-4o"
        assert len(d["detections"]) == 1
        assert d["reflections_injected"] == 1
        assert d["final_state"] == "open"
        assert d["kill_reason"] == "reflection_failed: tool_repeat"
        assert len(d["step_timeline"]) == 3

    def test_round_trip_via_from_dict(self) -> None:
        report = _build_report()
        restored = GuardReport._from_dict(report.to_dict())

        assert restored.total_steps == report.total_steps
        assert restored.total_tokens == report.total_tokens
        assert restored.estimated_cost_usd == pytest.approx(report.estimated_cost_usd)
        assert restored.model == report.model
        assert len(restored.detections) == len(report.detections)
        assert restored.detections[0].pattern == "tool_repeat"
        assert restored.reflections_injected == 1
        assert restored.final_state == BreakerState.OPEN
        assert restored.kill_reason == report.kill_reason

    def test_missing_optional_fields_ok(self) -> None:
        """Minimal dict without cost/model fields should not crash."""
        data = {
            "total_steps": 1,
            "total_tokens": 100,
            "detections": [],
            "reflections_injected": 0,
            "final_state": "closed",
            "kill_reason": None,
            "step_timeline": [],
        }
        r = GuardReport._from_dict(data)
        assert r.estimated_cost_usd is None
        assert r.model is None

    def test_unknown_final_state_graceful(self) -> None:
        data = {
            "total_steps": 1,
            "total_tokens": 50,
            "detections": [],
            "reflections_injected": 0,
            "final_state": "some_future_state_unknown",
            "kill_reason": None,
            "step_timeline": [],
        }
        r = GuardReport._from_dict(data)
        assert r.final_state is None  # warning logged, not raised


# ===========================================================================
# save() — JSON
# ===========================================================================


class TestSaveJSON:
    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.json"
        report.save(out)
        assert out.exists()

    def test_save_json_is_valid_json(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.json"
        report.save(out)
        data = json.loads(out.read_text())
        assert data["total_steps"] == 3

    def test_save_explicit_fmt_json(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "myreport.txt"  # unusual extension
        report.save(out, fmt="json")
        data = json.loads(out.read_text())
        assert "total_tokens" in data

    def test_save_json_roundtrip(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.json"
        report.save(out)
        restored = GuardReport.load(out)
        assert restored.total_steps == report.total_steps
        assert restored.estimated_cost_usd == pytest.approx(report.estimated_cost_usd)


# ===========================================================================
# save() — YAML
# ===========================================================================


class TestSaveYAML:
    def test_save_yaml_fallback_to_json_when_no_pyyaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When pyyaml is not installed, should fall back to JSON gracefully."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("mocked: pyyaml not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        report = _build_report()
        out = tmp_path / "report.yaml"
        report.save(out)

        # Should have fallen back to a .json file
        json_out = out.with_suffix(".json")
        assert json_out.exists()
        data = json.loads(json_out.read_text())
        assert data["total_steps"] == 3

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("yaml"),
        reason="pyyaml not installed",
    )
    def test_save_yaml_with_pyyaml(self, tmp_path: Path) -> None:
        import yaml

        report = _build_report()
        out = tmp_path / "report.yaml"
        report.save(out)
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert data["total_steps"] == 3

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("yaml"),
        reason="pyyaml not installed",
    )
    def test_save_yml_extension(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.yml"
        report.save(out)
        assert out.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("yaml"),
        reason="pyyaml not installed",
    )
    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.yaml"
        report.save(out)
        restored = GuardReport.load(out)
        assert restored.total_steps == report.total_steps
        assert restored.model == "gpt-4o"


# ===========================================================================
# load()
# ===========================================================================


class TestLoad:
    def test_load_json(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.json"
        report.save(out)
        loaded = GuardReport.load(out)
        assert loaded.total_steps == 3
        assert loaded.final_state == BreakerState.OPEN

    def test_load_nonexistent_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GuardReport.load(tmp_path / "nonexistent.json")

    def test_load_detections_preserved(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "report.json"
        report.save(out)
        loaded = GuardReport.load(out)
        assert len(loaded.detections) == 1
        assert loaded.detections[0].pattern == "tool_repeat"
        assert loaded.detections[0].confidence == pytest.approx(0.9)

    def test_load_unknown_extension_tries_json(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "myreport.data"
        out.write_text(report.to_json(), encoding="utf-8")
        loaded = GuardReport.load(out)
        assert loaded.total_tokens == 300

    def test_load_bad_unknown_extension_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "report.data"
        out.write_text("not json at all !!!", encoding="utf-8")
        with pytest.raises(ValueError):
            GuardReport.load(out)

    def test_load_string_path(self, tmp_path: Path) -> None:
        """load() should accept plain strings as well as Path objects."""
        report = _build_report()
        out = tmp_path / "report.json"
        report.save(str(out))
        loaded = GuardReport.load(str(out))
        assert loaded.model == "gpt-4o"
