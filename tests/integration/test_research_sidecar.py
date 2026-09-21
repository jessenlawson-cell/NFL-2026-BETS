from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_bets import research_sidecar
from nfl_bets.config import Settings


def test_capture_writes_only_immutable_research_artifacts(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    protected_spec = settings.root / "CHALLENGER_SPEC.md"
    protected_spec.write_text("frozen challenger 0.2", encoding="utf-8")
    protected_ledger = settings.root / "model_predictions.csv"
    protected_ledger.write_text("prediction_id\n", encoding="utf-8")
    frame = pl.DataFrame({"season": [2026], "week": [2], "value": [1.0]})

    monkeypatch.setattr(research_sidecar.nfl, "load_schedules", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_pbp", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_injuries", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_depth_charts", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_players", lambda: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_snap_counts", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_ftn_charting", lambda seasons: frame)
    monkeypatch.setattr(
        research_sidecar.nfl,
        "load_nextgen_stats",
        lambda seasons, stat_type: frame.with_columns(pl.lit(stat_type).alias("stat_type")),
    )

    result = research_sidecar.capture_research_snapshot(
        [2026],
        settings=settings,
        captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
    )

    capture_id = "20260921T120000000000Z"
    raw_root = settings.data_dir / "research" / "challenger_0_3" / "raw" / capture_id
    manifest_path = (
        settings.manifests_dir / "research" / "challenger_0_3" / f"{capture_id}.json"
    )
    report_path = settings.reports_dir / "research" / "challenger_0_3" / f"{capture_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result == {
        "capture_id": capture_id,
        "status": "RESEARCH_ONLY_UNWEIGHTED",
        "decision": "PASS",
        "datasets": 10,
        "rows": 10,
        "manifest": manifest_path.relative_to(settings.root).as_posix(),
        "report": report_path.relative_to(settings.root).as_posix(),
    }
    assert sorted(path.stem for path in raw_root.glob("*.parquet")) == [
        "depth_charts",
        "ftn_charting",
        "injuries",
        "ngs_passing",
        "ngs_receiving",
        "ngs_rushing",
        "pbp",
        "players",
        "schedules",
        "snap_counts",
    ]
    assert manifest["candidate_status"] == "RESEARCH_ONLY_UNWEIGHTED"
    assert manifest["decision"] == "PASS"
    assert manifest["operational_state_unchanged"] is True
    assert {
        entry["historical_availability_status"] for entry in manifest["datasets"].values()
    } == {"RESEARCH_BACKFILL_NOT_OOS_ELIGIBLE"}
    assert {
        entry["source_available_at_utc"] for entry in manifest["datasets"].values()
    } == {"2026-09-21T12:00:00Z"}
    assert all(entry["schema_hash"] for entry in manifest["datasets"].values())
    assert all(entry["content_hash"] for entry in manifest["datasets"].values())
    assert manifest["personnel_coverage"]["status"] == "COVERAGE_ONLY_UNWEIGHTED"
    assert manifest["personnel_coverage"]["pass_rusher_role_status"] == (
        "UNDEFINED_RESEARCH_DIAGNOSTIC"
    )
    assert protected_spec.read_text(encoding="utf-8") == "frozen challenger 0.2"
    assert protected_ledger.read_text(encoding="utf-8") == "prediction_id\n"
    assert report_path.exists()


def test_capture_fails_closed_if_a_loader_changes_operational_state(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    protected_spec = settings.root / "CHALLENGER_SPEC.md"
    protected_spec.write_text("frozen challenger 0.2", encoding="utf-8")
    frame = pl.DataFrame({"season": [2026], "week": [2], "value": [1.0]})

    def mutating_schedule_loader(seasons: list[int]) -> pl.DataFrame:
        protected_spec.write_text("mutated", encoding="utf-8")
        return frame

    monkeypatch.setattr(research_sidecar.nfl, "load_schedules", mutating_schedule_loader)
    monkeypatch.setattr(research_sidecar.nfl, "load_pbp", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_injuries", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_depth_charts", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_players", lambda: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_snap_counts", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_ftn_charting", lambda seasons: frame)
    monkeypatch.setattr(
        research_sidecar.nfl,
        "load_nextgen_stats",
        lambda seasons, stat_type: frame.with_columns(pl.lit(stat_type).alias("stat_type")),
    )

    with pytest.raises(RuntimeError, match="changed protected operational state"):
        research_sidecar.capture_research_snapshot(
            [2026],
            settings=settings,
            captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        )

    research_root = settings.data_dir / "research" / "challenger_0_3"
    assert not (research_root / "raw").exists()
    assert not (settings.manifests_dir / "research").exists()


def test_diagnostic_schema_failure_does_not_promote_raw_capture(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    frame = pl.DataFrame({"season": [2026], "week": [2], "value": [1.0]})
    snaps = pl.DataFrame({"pfr_player_id": [1], "position": ["QB"]})
    players = pl.DataFrame({"pfr_id": ["1"], "gsis_id": ["player-gsis"]})

    monkeypatch.setattr(research_sidecar.nfl, "load_schedules", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_pbp", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_injuries", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_depth_charts", lambda seasons: frame)
    monkeypatch.setattr(research_sidecar.nfl, "load_players", lambda: players)
    monkeypatch.setattr(research_sidecar.nfl, "load_snap_counts", lambda seasons: snaps)
    monkeypatch.setattr(research_sidecar.nfl, "load_ftn_charting", lambda seasons: frame)
    monkeypatch.setattr(
        research_sidecar.nfl,
        "load_nextgen_stats",
        lambda seasons, stat_type: frame.with_columns(pl.lit(stat_type).alias("stat_type")),
    )

    with pytest.raises(pl.exceptions.SchemaError):
        research_sidecar.capture_research_snapshot(
            [2026],
            settings=settings,
            captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        )

    final_root = (
        settings.data_dir
        / "research"
        / "challenger_0_3"
        / "raw"
        / "20260921T120000000000Z"
    )
    assert not final_root.exists()
