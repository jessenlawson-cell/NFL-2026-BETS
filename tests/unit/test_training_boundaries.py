from __future__ import annotations

import polars as pl

from nfl_bets.model import training


def _nested_rows() -> pl.DataFrame:
    rows = []
    for season in (2021, 2022, 2023):
        for index in range(12):
            margin = float(index - 6)
            total = float(40 + index)
            rows.append(
                {
                    "season": season,
                    "target_margin": margin,
                    "target_total": total,
                    "spread_final": margin + 0.25,
                    "spread_line": margin + 0.5,
                    "total_final": total - 0.25,
                    "total_line": total - 0.5,
                }
            )
    return pl.DataFrame(rows)


def test_block_bootstrap_is_reproducible(monkeypatch) -> None:
    monkeypatch.setattr(training, "BOOTSTRAP_SAMPLES", 20)
    first = training._bootstrap(_nested_rows())
    second = training._bootstrap(_nested_rows())
    assert first == second


def test_development_hash_ignores_rows_outside_supplied_matrix() -> None:
    config = training.FeatureConfig(2.0, 1.0)
    columns = [
        "game_id",
        "season",
        *training.FEATURE_COLUMNS,
        "target_margin",
        "target_total",
        "spread_line",
        "total_line",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    ]
    row = {column: 0.0 for column in columns}
    row.update(game_id="2024_01_A_B", season=2024)
    development = pl.DataFrame([row])
    assert training._development_hash(development, config) == training._development_hash(
        development.clone(), config
    )
