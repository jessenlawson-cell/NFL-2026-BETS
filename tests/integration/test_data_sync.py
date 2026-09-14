from __future__ import annotations

import json

import polars as pl
import pytest

from nfl_bets.config import Settings
from nfl_bets.data import sync as sync_module
from nfl_bets.db import connect
from nfl_bets.validation import validate_all


def test_sync_uses_nflreadpy_loaders_and_writes_authoritative_artifacts(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    schedules = pl.DataFrame(
        [
            {
                "game_id": "2025_01_NYJ_BUF",
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "gameday": "2025-09-07",
                "gametime": "13:00",
                "away_team": "NYJ",
                "home_team": "BUF",
                "away_score": 17,
                "home_score": 24,
                "result": 7,
                "total": 41,
                "away_rest": 7,
                "home_rest": 7,
                "spread_line": 6.5,
                "total_line": 44.5,
                "away_spread_odds": -110,
                "home_spread_odds": -110,
                "under_odds": -110,
                "over_odds": -110,
            }
        ]
    )
    injuries = pl.DataFrame(
        [
            {
                "season": 2025,
                "season_type": "REG",
                "week": 1,
                "team": "BUF",
                "gsis_id": "player-1",
                "full_name": "Fixture Player",
                "position": "QB",
                "report_status": "Questionable",
                "practice_status": "Limited",
                "report_primary_injury": "Ankle",
                "report_secondary_injury": None,
            }
        ]
    )
    snaps = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "game_id": "2025_01_NYJ_BUF",
                "player": "Fixture Player",
                "pfr_player_id": "Fixture00",
                "position": "QB",
                "team": "BUF",
                "offense_snaps": 60,
                "offense_pct": 1.0,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 0,
                "st_pct": 0.0,
            }
        ]
    )
    team_stats = pl.DataFrame([{"season": 2025, "week": 1, "team": "BUF"}])
    monkeypatch.setattr(sync_module.nfl, "load_schedules", lambda seasons: schedules)
    monkeypatch.setattr(
        sync_module.nfl, "load_team_stats", lambda seasons, summary_level: team_stats
    )
    monkeypatch.setattr(sync_module.nfl, "load_injuries", lambda seasons: injuries)
    monkeypatch.setattr(sync_module.nfl, "load_snap_counts", lambda seasons: snaps)

    result = sync_module.sync_data(2025, 2025, include_pbp=False, settings=settings)

    assert result == {"schedules": 1, "team_stats": 1, "injuries": 1, "snap_counts": 1}
    games = pl.read_csv(settings.root / "games.csv")
    assert games.height == 1
    assert games["source"][0] == "nflverse:nflreadpy"
    assert games["content_hash"][0]
    assert not list((settings.data_dir / "curated").glob("*.csv"))
    manifest = json.loads(
        (settings.manifests_dir / "nflverse_sync.latest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "COMPLETE"
    assert manifest["source_update_status"] == "UNAVAILABLE_FROM_PROVIDER_LIBRARY"
    first_hash = games["content_hash"][0]
    sync_module.sync_data(2025, 2025, include_pbp=False, settings=settings)
    assert pl.read_csv(settings.root / "games.csv")["content_hash"][0] == first_hash
    assert validate_all(settings)["status"] == "VALID"
    with connect(settings) as connection:
        assert connection.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert {
            row[0] for row in connection.execute("SELECT status FROM ingestion_runs")
        } == {"COMPLETE"}


def test_required_failure_is_logged_without_promoting_partial_files(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    monkeypatch.setattr(
        sync_module.nfl,
        "load_schedules",
        lambda seasons: (_ for _ in ()).throw(RuntimeError("fixture network failure")),
    )
    with pytest.raises(RuntimeError, match="Required schedules failed"):
        sync_module.sync_data(2025, 2025, include_pbp=False, settings=settings)
    assert not (settings.root / "games.csv").exists()
    failed = list(settings.manifests_dir.glob("nflverse_sync_*.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text(encoding="utf-8"))["status"] == "FAILED"
