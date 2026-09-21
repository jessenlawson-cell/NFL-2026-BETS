from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database, transaction
from nfl_bets.model.residuals import OutcomeProbabilities
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
    monkeypatch,
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
        ("spreads", 2.5, 0.60, "spread-prediction"),
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
            "INSERT INTO raw_snapshots("
            "snapshot_id,provider,kind,snapshot_purpose,path,headers_path,retrieved_at_utc,"
            "content_hash,byte_count) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "snapshot",
                "fixture",
                "full-board",
                "DECISION",
                "fixture.json",
                None,
                "2026-09-20T16:00:00Z",
                "raw-hash",
                1,
            ),
        )
        connection.execute(
            "INSERT INTO raw_snapshots("
            "snapshot_id,provider,kind,snapshot_purpose,path,headers_path,retrieved_at_utc,"
            "content_hash,byte_count) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "close-snapshot",
                "fixture",
                "full-board",
                "CLOSE",
                "close.json",
                None,
                "2026-09-20T16:45:00Z",
                "close-hash",
                1,
            ),
        )
        columns = ARTIFACT_SCHEMAS["model_predictions"].columns
        connection.executemany(
            f"INSERT INTO model_predictions ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            [tuple(row[column] for column in columns) for row in predictions],
        )
        close_quotes = []
        for selection, price, probability in (("HOME", -105, 0.52), ("AWAY", -115, 0.48)):
            close_quotes.append(
                _record(
                    "market_odds",
                    snapshot_id="close-snapshot",
                    provider_event_id="event",
                    game_id=game["game_id"],
                    commence_time_utc=game["kickoff_utc"],
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    bookmaker_group="anchor",
                    market="spreads",
                    selection=selection,
                    point=-3.5 if selection == "HOME" else 3.5,
                    canonical_line=3.5,
                    american_price=price,
                    decimal_price=1.95,
                    implied_probability=0.53,
                    vig_free_probability=probability,
                    overround=1.05,
                    last_update_utc="2026-09-20T16:40:00Z",
                    source="fixture",
                    retrieved_at_utc="2026-09-20T16:45:00Z",
                    source_updated_at_utc="2026-09-20T16:40:00Z",
                    schema_version="1.1.0",
                )
            )
        quote_columns = ARTIFACT_SCHEMAS["market_odds"].columns
        connection.executemany(
            f"INSERT INTO market_odds ({','.join(quote_columns)}) "
            f"VALUES ({','.join('?' for _ in quote_columns)})",
            [tuple(row[column] for column in quote_columns) for row in close_quotes],
        )

    class Mapper:
        def probabilities(
            self, projection: float, line: float, stratum: float
        ) -> OutcomeProbabilities:
            if line == projection:
                return OutcomeProbabilities(win=0.45, push=0.10, loss=0.45)
            return OutcomeProbabilities(win=0.65, push=0.05, loss=0.30)

    monkeypatch.setattr(
        "nfl_bets.prospective.load_frozen_bundle",
        lambda *_, **__: SimpleNamespace(candidate=SimpleNamespace(spread_residuals=Mapper())),
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
    assert rows["spreads"]["decision_snapshot_id"] == "snapshot"
    assert rows["spreads"]["closing_snapshot_id"] == "close-snapshot"
    assert rows["spreads"]["clv_status"] == "AVAILABLE"
    assert rows["spreads"]["market_probability_movement"] == pytest.approx(0.02)
    assert rows["spreads"]["decision_price_clv"] is not None
    assert rows["spreads"]["line_clv"] == 1.0
    assert rows["spreads"]["key_numbers_crossed_json"] == "[3]"
    assert rows["spreads"]["closing_contract_ev"] is not None
    assert rows["totals"]["result"] == "PUSH"
    assert rows["totals"]["eligible_non_push"] == 0
    assert rows["totals"]["model_brier"] is None
    assert rows["totals"]["model_log_loss"] is None
    assert rows["totals"]["clv_status"] == "UNAVAILABLE_MISSING_CLOSE"
    assert rows["totals"]["line_clv"] is None
