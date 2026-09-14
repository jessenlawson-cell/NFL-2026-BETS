from __future__ import annotations

import json
from datetime import UTC, datetime

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database, transaction
from nfl_bets.prospective import settle_predictions
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import canonical_hash


def _record(schema_name: str, **values: object) -> dict[str, object]:
    record: dict[str, object] = {column: None for column in ARTIFACT_SCHEMAS[schema_name].columns}
    record.update(values)
    record["content_hash"] = canonical_hash(record)
    return record


def test_settlement_keeps_pushes_out_of_probability_scores_and_is_idempotent(
    tmp_path,
) -> None:
    settings = Settings.for_root(tmp_path)
    settings.ensure_directories()
    initialize_database(settings)
    (settings.manifests_dir / "prospective_policy_1.1.2.json").write_text(
        json.dumps(
            {
                "model_version": "1.1.2",
                "test_season": 2026,
                "prospective_start_utc": "2026-09-14T01:53:59Z",
                "selection_rule_version": "1.0.0",
                "close_window_minutes_before_kickoff": {"earliest": 90, "latest": 5},
                "quote_max_age_minutes": 30,
            }
        ),
        encoding="utf-8",
    )
    game = _record(
        "games",
        game_id="2026_02_BUF_NYJ",
        season=2026,
        week=2,
        game_type="REG",
        kickoff_utc="2026-09-20T17:00:00Z",
        away_team="BUF",
        home_team="NYJ",
        away_score=20,
        home_score=24,
        source="fixture",
        retrieved_at_utc="2026-09-21T12:00:00Z",
        schema_version="1.1.0",
    )
    predictions = []
    for market, line, model_probability, prediction_id in (
        ("spreads", 3.0, 0.60, "spread-prediction"),
        ("totals", 44.0, 0.55, "total-prediction"),
    ):
        push_probability = 0.05
        predictions.append(
            _record(
                "model_predictions",
                prediction_id=prediction_id,
                model_version="1.1.2",
                model_artifact_hash="artifact",
                model_spec_hash="spec",
                policy_hash="policy",
                git_commit="commit",
                snapshot_id="snapshot",
                provider_event_id="event",
                game_id=game["game_id"],
                season=2026,
                week=2,
                kickoff_utc=game["kickoff_utc"],
                market=market,
                orientation="HOME" if market == "spreads" else "OVER",
                prediction_created_at_utc="2026-09-20T16:01:00Z",
                snapshot_retrieved_at_utc="2026-09-20T16:00:00Z",
                feature_as_of_utc="2026-09-20T15:00:00Z",
                feature_input_hash="features",
                feature_row_hash="row",
                pinnacle_updated_at_utc="2026-09-20T15:55:00Z",
                pinnacle_line=line,
                pinnacle_orientation_price=-110,
                pinnacle_other_price=-110,
                pinnacle_orientation_no_vig_probability=0.5,
                pinnacle_overround=1.0476,
                retail_books_count=0,
                raw_adjustment=1.0,
                adjustment_weight=0.1,
                final_projection=line + 0.1,
                raw_non_push_win_probability=model_probability,
                calibrated_non_push_win_probability=model_probability,
                model_win_probability=(1 - push_probability) * model_probability,
                model_push_probability=push_probability,
                model_loss_probability=(1 - push_probability) * (1 - model_probability),
                uncertainty_status="UNAVAILABLE_FOR_BETTING",
                eligibility_status="ELIGIBLE_CLOSE_PROXY",
                decision="PASS",
                pass_reason="observer",
                source="fixture",
                retrieved_at_utc="2026-09-20T16:01:00Z",
                schema_version="1.1.0",
            )
        )
    with transaction(settings) as connection:
        game_columns = ARTIFACT_SCHEMAS["games"].columns
        connection.execute(
            f"INSERT INTO games ({','.join(game_columns)}) "
            f"VALUES ({','.join('?' for _ in game_columns)})",
            tuple(game[column] for column in game_columns),
        )
        connection.execute(
            "INSERT INTO raw_snapshots VALUES (?,?,?,?,?,?,?,?)",
            (
                "snapshot",
                "fixture",
                "full-board",
                "fixture.json",
                None,
                "2026-09-20T16:00:00Z",
                "raw-hash",
                1,
            ),
        )
        columns = ARTIFACT_SCHEMAS["model_predictions"].columns
        connection.executemany(
            f"INSERT INTO model_predictions ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            [tuple(row[column] for column in columns) for row in predictions],
        )
    first = settle_predictions(
        datetime(2026, 9, 21, tzinfo=UTC),
        settings=settings,
        settled_at=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
    )
    second = settle_predictions(
        datetime(2026, 9, 21, tzinfo=UTC),
        settings=settings,
        settled_at=datetime(2026, 9, 21, 12, 5, tzinfo=UTC),
    )
    assert first["inserted"] == 2
    assert second["unchanged"] == 2
    with connect(settings) as connection:
        rows = {
            row["market"]: dict(row)
            for row in connection.execute("SELECT * FROM prospective_evaluations")
        }
    assert rows["spreads"]["result"] == "WIN"
    assert rows["spreads"]["eligible_non_push"] == 1
    assert rows["spreads"]["model_brier"] is not None
    assert rows["totals"]["result"] == "PUSH"
    assert rows["totals"]["eligible_non_push"] == 0
    assert rows["totals"]["model_brier"] is None
    assert rows["totals"]["model_log_loss"] is None
