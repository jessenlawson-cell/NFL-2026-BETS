from __future__ import annotations

import math

import polars as pl
import pytest

from nfl_bets.config import Settings
from nfl_bets.features.build import METRICS
from nfl_bets.model.training import test_model as run_model_test
from nfl_bets.model.training import train_model


def _write_synthetic_history(settings: Settings) -> None:
    games: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    for season in range(2010, 2026):
        games_in_season = 101 if season == 2025 else 20
        for index in range(games_in_season):
            game_id = f"{season}_{index:03d}_AWY_HME"
            margin = float((index % 13) - 6)
            total = float(41 + (index % 9))
            spread_win = index % 2 == 0
            total_win = index % 3 != 0
            spread_line = margin - 0.5 if spread_win else margin + 0.5
            total_line = total - 0.5 if total_win else total + 0.5
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": index // 16 + 1,
                    "game_type": "REG",
                    "kickoff_utc": f"{season}-09-01T12:00:00Z",
                    "away_team": "AWY",
                    "home_team": "HME",
                    "away_score": 20.0,
                    "home_score": 20.0 + margin,
                    "home_rest": 7.0,
                    "away_rest": 7.0,
                    "spread_line": spread_line,
                    "total_line": total_line,
                    "home_spread_odds": -10000 if spread_win else 10000,
                    "away_spread_odds": 10000 if spread_win else -10000,
                    "over_odds": -10000 if total_win else 10000,
                    "under_odds": 10000 if total_win else -10000,
                }
            )
            for is_home, team in ((1, "HME"), (0, "AWY")):
                signal = math.sin(index + season * 0.01 + is_home)
                row = {
                    "game_id": game_id,
                    "season": season,
                    "is_home": is_home,
                    "team_id": team,
                }
                row.update(
                    {metric: signal + offset * 0.001 for offset, metric in enumerate(METRICS)}
                )
                teams.append(row)
    pl.DataFrame(games).write_csv(settings.root / "games.csv")
    pl.DataFrame(teams).write_csv(settings.root / "team_metrics.csv")


def test_untouched_gate_defaults_to_pass_only_and_cannot_be_repeated(tmp_path) -> None:
    settings = Settings(root=tmp_path)
    _write_synthetic_history(settings)
    train_model("fixture-v1", settings)
    report = run_model_test("fixture-v1", 2025, settings)
    assert report["status"] == "PASS_ONLY"
    with pytest.raises(RuntimeError, match="already consumed"):
        run_model_test("fixture-v1", 2025, settings)
