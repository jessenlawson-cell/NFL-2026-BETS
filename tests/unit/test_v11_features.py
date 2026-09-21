from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_bets.features.v11 import (
    _pressure_and_neutral_observations,
    _snap_continuity,
)


def test_pressure_and_neutral_rush_proxies_are_play_derived() -> None:
    rows = [
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": 0.2,
            "success": 1.0,
            "qb_dropback": 1.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 1.0,
            "qtr": 1.0,
            "score_differential": 0.0,
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": -0.4,
            "success": 0.0,
            "qb_dropback": 1.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 1.0,
            "qb_hit": 1.0,
            "qtr": 2.0,
            "score_differential": 3.0,
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": 0.3,
            "success": 1.0,
            "qb_dropback": 0.0,
            "rush_attempt": 1.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "qtr": 3.0,
            "score_differential": -7.0,
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "B",
            "defteam": "A",
            "epa": 0.1,
            "success": 1.0,
            "qb_dropback": 1.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "qtr": 1.0,
            "score_differential": 0.0,
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "B",
            "defteam": "A",
            "epa": -0.2,
            "success": 0.0,
            "qb_dropback": 0.0,
            "rush_attempt": 1.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "qtr": 4.0,
            "score_differential": 0.0,
        },
    ]
    result = _pressure_and_neutral_observations(pl.DataFrame(rows))
    team_a = result.filter(pl.col("team_id") == "A").row(0, named=True)
    assert team_a["off_sack_rate"] == pytest.approx(0.5)
    assert team_a["off_qb_hit_rate"] == pytest.approx(1.0)
    assert team_a["off_neutral_rush_epa"] == pytest.approx(0.3)
    assert team_a["off_neutral_rush_success_rate"] == pytest.approx(1.0)


def test_snap_continuity_uses_immediately_previous_team_game() -> None:
    games = pl.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "kickoff_utc": "2025-09-01T17:00:00Z",
                "home_team": "A",
                "away_team": "B",
            },
            {
                "game_id": "2025_02_C_A",
                "season": 2025,
                "week": 2,
                "game_type": "REG",
                "kickoff_utc": "2025-09-08T17:00:00Z",
                "home_team": "C",
                "away_team": "A",
            },
        ]
    )
    usage = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": week,
                "game_id": game,
                "team_id": "A",
                "player_id": player,
                "offense_pct": offense,
                "defense_pct": defense,
            }
            for week, game, player, offense, defense in (
                (1, "2025_01_A_B", "returning", 0.5, 0.6),
                (1, "2025_01_A_B", "departed", 0.5, 0.4),
                (2, "2025_02_C_A", "returning", 0.5, 0.6),
                (2, "2025_02_C_A", "new", 0.5, 0.4),
            )
        ]
    )
    result = _snap_continuity(usage, games).sort("week")
    assert result["off_snap_continuity"][0] is None
    assert result["def_snap_continuity"][0] is None
    assert result["off_snap_continuity"][1] == pytest.approx(0.5)
    assert result["def_snap_continuity"][1] == pytest.approx(0.6)


def test_feature_schedule_excludes_started_but_incomplete_games() -> None:
    from nfl_bets.features.build import _team_schedule

    games = pl.DataFrame(
        [
            {
                "game_id": "2026_01_A_B",
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "kickoff_utc": "2026-09-01T17:00:00Z",
                "away_team": "A",
                "home_team": "B",
                "away_score": 20.0,
                "home_score": 21.0,
            },
            {
                "game_id": "2026_01_C_D",
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "kickoff_utc": "2026-09-02T17:00:00Z",
                "away_team": "C",
                "home_team": "D",
                "away_score": None,
                "home_score": None,
            },
            {
                "game_id": "2026_02_C_E",
                "season": 2026,
                "week": 2,
                "game_type": "REG",
                "kickoff_utc": "2026-09-09T17:00:00Z",
                "away_team": "C",
                "home_team": "E",
                "away_score": None,
                "home_score": None,
            },
        ]
    )
    result = _team_schedule(games, datetime(2026, 9, 3, tzinfo=UTC))
    assert set(result["game_id"].unique()) == {"2026_01_A_B", "2026_02_C_E"}
