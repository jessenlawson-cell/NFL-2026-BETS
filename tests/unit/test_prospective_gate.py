from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.prospective import run_prospective_test
from nfl_bets.schemas import ARTIFACT_SCHEMAS


def _policy(settings: Settings) -> None:
    settings.ensure_directories()
    (settings.manifests_dir / "prospective_policy_1.1.2.json").write_text(
        json.dumps(
            {
                "model_version": "1.1.2",
                "test_season": 2026,
                "prospective_start_utc": "2026-09-14T01:53:59Z",
                "formal_test_minimum_week": 8,
                "minimum_non_push_observations_per_market": 100,
                "spec_hash": "spec",
            }
        ),
        encoding="utf-8",
    )


def _empty_artifacts(settings: Settings) -> None:
    for name, schema in ARTIFACT_SCHEMAS.items():
        pl.DataFrame({column: [] for column in schema.columns}).write_csv(
            settings.root / f"{name}.csv"
        )


def test_week_eight_test_defers_without_consuming_small_sample(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    _policy(settings)
    _empty_artifacts(settings)
    result = run_prospective_test(8, settings=settings)
    assert result["status"] == "DEFERRED_INSUFFICIENT_SAMPLE"
    assert result["formal_test_consumed"] is False
    with connect(settings) as connection:
        registry = connection.execute("SELECT * FROM prospective_test_registry").fetchone()
    assert registry["status"] == "DEFERRED_INSUFFICIENT_SAMPLE"
    assert registry["started_at_utc"] is None


def test_formal_test_is_consumed_exactly_once_after_minimum_sample(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    _policy(settings)
    _empty_artifacts(settings)
    games: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    for index in range(100):
        week = index % 8 + 1
        game_id = f"2026_{week:02d}_GAME_{index:03d}"
        kickoff = datetime(2026, 9, 21, 17, 0, tzinfo=UTC).isoformat()
        win = index % 2 == 0
        home_score, away_score = (25, 20) if win else (20, 23)
        games.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "game_type": "REG",
                "kickoff_utc": kickoff,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
        for market in ("spreads", "totals"):
            model_probability = 0.60 if win else 0.40
            market_probability = 0.55 if win else 0.45
            evaluations.append(
                {
                    "model_version": "1.1.2",
                    "season": 2026,
                    "week": week,
                    "kickoff_utc": kickoff,
                    "game_id": game_id,
                    "market": market,
                    "eligible_non_push": 1,
                    "result": "WIN" if win else "LOSS",
                    "model_non_push_win_probability": model_probability,
                    "decision_market_probability": market_probability,
                    "market_probability_movement": None,
                    "final_projection": 1.0,
                    "projection_error": 0.0,
                    "decision_price_clv": None,
                    "decision_line_clv": None,
                    "line_clv": None,
                    "closing_contract_ev": None,
                    "clv_status": "UNAVAILABLE_MISSING_CLOSE",
                    "exclusion_reason": None,
                }
            )
    pl.DataFrame(games).write_csv(settings.root / "games.csv")
    pl.DataFrame(evaluations).write_csv(settings.root / "prospective_evaluations.csv")
    (settings.root / "PROJECT_STATE.md").write_text(
        "# State\n- **Last Model Version:** old\n- **System Status:** old\n"
        "- **Betting Status:** PASS-only\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("nfl_bets.validation.validate_all", lambda settings: {"status": "VALID"})
    result = run_prospective_test(8, settings=settings)
    assert result["formal_test_consumed"] is True
    assert result["decision"] == "PASS"
    with pytest.raises(RuntimeError, match="already been consumed"):
        run_prospective_test(8, settings=settings)
