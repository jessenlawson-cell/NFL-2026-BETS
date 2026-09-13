from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_bets.features.build import METRICS, build_team_metric_matrix


def _team_games(values: list[float]) -> pl.DataFrame:
    rows = []
    for index, value in enumerate(values, start=1):
        row = {
            "game_id": f"2020_{index:02d}_A_B",
            "team_id": "A",
            "season": 2020,
            "week": index,
            "kickoff_dt": datetime(2020, 9, 1, tzinfo=UTC) + timedelta(days=index),
            "opponent_team_id": "B",
            "is_home": True,
        }
        row.update({metric: value for metric in METRICS})
        rows.append(row)
    return pl.DataFrame(rows)


def test_ewma_uses_shift_one_before_weighting() -> None:
    result = build_team_metric_matrix(_team_games([1.0, 2.0, 3.0, 4.0, 5.0]), half_life=2.0)
    rows = result.sort("week").to_dicts()
    assert rows[0]["off_pass_epa"] is None
    expected_week_3 = (2.0 + (2**-0.5) * 1.0) / (1.0 + 2**-0.5)
    assert rows[2]["off_pass_epa"] == pytest.approx(expected_week_3)
    assert rows[4]["games_available"] == 4


def test_current_game_mutation_cannot_change_its_pregame_feature() -> None:
    original = build_team_metric_matrix(_team_games([1.0, 2.0, 3.0, 4.0]), 2.0)
    mutated = build_team_metric_matrix(_team_games([1.0, 2.0, 9999.0, 4.0]), 2.0)
    original_week_three = original.filter(pl.col("week") == 3)["off_pass_epa"][0]
    mutated_week_three = mutated.filter(pl.col("week") == 3)["off_pass_epa"][0]
    assert mutated_week_three == pytest.approx(original_week_three)
