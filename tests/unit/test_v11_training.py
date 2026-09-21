from __future__ import annotations

import polars as pl
import pytest

from nfl_bets.config import Settings
from nfl_bets.features.v11 import V11_METRICS
from nfl_bets.model.v11 import _adjustment_weight, assemble_v11_matrix


def test_v11_adjustment_weight_is_fit_against_market_residual() -> None:
    frame = pl.DataFrame(
        {
            "raw_adjustment": [1.0, -2.0, 3.0],
            "actual": [11.0, 8.0, 13.0],
            "market_projection": [10.0, 10.0, 10.0],
        }
    )
    assert _adjustment_weight(frame) == pytest.approx(1.0)


def test_v11_matrix_builds_directional_and_total_matchups(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    games = pl.DataFrame(
        [
            {
                "game_id": "2014_01_A_B",
                "season": 2014,
                "week": 1,
                "game_type": "REG",
                "kickoff_utc": "2014-09-01T17:00:00Z",
                "away_team": "A",
                "home_team": "B",
                "away_score": 20.0,
                "home_score": 24.0,
                "home_rest": 7.0,
                "away_rest": 6.0,
                "spread_line": 3.0,
                "total_line": 43.0,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
            }
        ]
    )
    games.write_csv(settings.root / "games.csv")
    rows = []
    for is_home, team, value in ((True, "B", 2.0), (False, "A", 1.0)):
        row: dict[str, object] = {
            "game_id": "2014_01_A_B",
            "season": 2014,
            "is_home": is_home,
            "team_id": team,
        }
        row.update({metric: value for metric in V11_METRICS})
        rows.append(row)
    matrix = assemble_v11_matrix(pl.DataFrame(rows), settings, maximum_season=2014)
    assert matrix["pass_epa_matchup_difference"][0] == pytest.approx(0.0)
    assert matrix["pass_epa_matchup_sum"][0] == pytest.approx(6.0)
    assert matrix["rest_difference"][0] == pytest.approx(1.0)


def test_v11_development_refuses_2026_outcomes(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    with pytest.raises(RuntimeError, match="cannot read 2026 outcomes"):
        assemble_v11_matrix(pl.DataFrame(), settings, maximum_season=2026)
