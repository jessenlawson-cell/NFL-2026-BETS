from __future__ import annotations

import polars as pl

from nfl_bets.config import Settings
from nfl_bets.data import sync as sync_module
from nfl_bets.db import connect


def test_sync_uses_nflreadpy_loaders_and_writes_authoritative_artifacts(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(root=tmp_path)
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
    with connect(settings) as connection:
        assert connection.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert connection.execute("SELECT status FROM ingestion_runs").fetchone()[0] == "COMPLETE"
