from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database, transaction
from nfl_bets.features.build import build_lag_matrix
from nfl_bets.features.v11 import V11_METRICS
from nfl_bets.odds.client import _persist_parsed
from nfl_bets.odds.consensus import parse_board
from nfl_bets.prospective import predict_snapshot
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_text, canonical_hash, sha256_bytes


def _game_record() -> dict[str, object]:
    record: dict[str, object] = {column: None for column in ARTIFACT_SCHEMAS["games"].columns}
    record.update(
        {
            "game_id": "2026_02_BUF_NYJ",
            "season": 2026,
            "week": 2,
            "game_type": "REG",
            "kickoff_utc": "2026-09-20T17:00:00Z",
            "away_team": "BUF",
            "home_team": "NYJ",
            "away_rest": 7,
            "home_rest": 7,
            "source": "fixture",
            "retrieved_at_utc": "2026-09-20T15:00:00Z",
            "schema_version": "1.1.0",
        }
    )
    record["content_hash"] = canonical_hash(record)
    return record


def _install_frozen_candidate(settings: Settings) -> None:
    repository = Path(__file__).resolve().parents[2]
    shutil.copy(repository / "MODEL_SPEC_V1_1.md", settings.root / "MODEL_SPEC_V1_1.md")
    source_artifact = repository / "artifacts" / "models" / "1.1.2"
    target_artifact = settings.artifacts_dir / "models" / "1.1.2"
    shutil.copytree(source_artifact, target_artifact)
    manifest = json.loads(
        (repository / "manifests" / "model_1.1.2_development.json").read_text(encoding="utf-8")
    )
    atomic_write_text(
        settings.manifests_dir / "model_1.1.2_development.json",
        json.dumps(manifest),
    )
    policy = json.loads(
        (repository / "manifests" / "prospective_policy_1.1.2.json").read_text(encoding="utf-8")
    )
    atomic_write_text(
        settings.manifests_dir / "prospective_policy_1.1.2.json",
        json.dumps(policy),
    )


def _install_features(settings: Settings) -> None:
    rows: list[dict[str, object]] = []
    for team, opponent, is_home in (("BUF", "NYJ", False), ("NYJ", "BUF", True)):
        prior = {
            "game_id": f"2025_PRIOR_{team}",
            "season": 2025,
            "week": 18,
            "kickoff_dt": datetime(2026, 1, 1, tzinfo=UTC),
            "team_id": team,
            "opponent_team_id": opponent,
            "is_home": is_home,
            **{metric: 0.1 for metric in V11_METRICS},
        }
        target = {
            "game_id": "2026_02_BUF_NYJ",
            "season": 2026,
            "week": 2,
            "kickoff_dt": datetime(2026, 9, 20, 17, 0, tzinfo=UTC),
            "team_id": team,
            "opponent_team_id": opponent,
            "is_home": is_home,
            **{metric: None for metric in V11_METRICS},
        }
        rows.extend([prior, target])
    lags = build_lag_matrix(pl.DataFrame(rows), V11_METRICS)
    lag_path = settings.runtime_dir / "features" / "v11_team_game_lags.parquet"
    lag_path.parent.mkdir(parents=True, exist_ok=True)
    lags.write_parquet(lag_path)
    manifest = {
        "artifact": "data/runtime/features/v11_team_game_lags.parquet",
        "as_of_utc": "2026-09-20T15:00:00Z",
        "content_hash": sha256_bytes(lag_path.read_bytes()),
        "injury_model_status": "DISABLED_INSUFFICIENT_HISTORICAL_COVERAGE",
    }
    atomic_write_text(
        settings.manifests_dir / "v11_feature_inputs.latest.json", json.dumps(manifest)
    )


def test_snapshot_prediction_is_pass_only_and_idempotent(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    settings.ensure_directories()
    initialize_database(settings)
    _install_frozen_candidate(settings)
    _install_features(settings)
    game = _game_record()
    pl.DataFrame([game]).select(ARTIFACT_SCHEMAS["games"].columns).write_csv(
        settings.root / "games.csv"
    )
    with transaction(settings) as connection:
        columns = ARTIFACT_SCHEMAS["games"].columns
        connection.execute(
            f"INSERT INTO games ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(game[column] for column in columns),
        )
        connection.execute(
            "INSERT INTO raw_snapshots VALUES (?,?,?,?,?,?,?,?)",
            (
                "snapshot",
                "the-odds-api",
                "full-board",
                "fixture.json",
                "fixture.headers.json",
                "2026-09-20T16:00:00Z",
                "fixture-hash",
                1,
            ),
        )
        connection.execute(
            "INSERT INTO api_requests(request_id,slot,request_kind,week_bucket,started_at_utc,"
            "completed_at_utc,status,raw_snapshot_id) VALUES (?,?,?,?,?,?,?,?)",
            (
                "request",
                "manual",
                "full-board",
                "2026-09-15",
                "2026-09-20T16:00:00Z",
                "2026-09-20T16:00:00Z",
                "COMPLETE",
                "snapshot",
            ),
        )
    board = [
        {
            "id": "event",
            "commence_time": "2026-09-20T17:00:00Z",
            "home_team": "New York Jets",
            "away_team": "Buffalo Bills",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "last_update": "2026-09-20T15:55:00Z",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "New York Jets", "point": -3.0, "price": -110},
                                {"name": "Buffalo Bills", "point": 3.0, "price": -110},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 44.0, "price": -110},
                                {"name": "Under", "point": 44.0, "price": -110},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    parsed = parse_board(
        board,
        "snapshot",
        "2026-09-20T16:00:00Z",
        settings.schema_version,
        {"event": "2026_02_BUF_NYJ"},
        settings.odds_freshness_minutes,
    )
    _persist_parsed(parsed, settings)
    first = predict_snapshot(
        "snapshot",
        settings=settings,
        prediction_time=datetime(2026, 9, 20, 16, 1, tzinfo=UTC),
        verify_git=False,
    )
    second = predict_snapshot(
        "snapshot",
        settings=settings,
        prediction_time=datetime(2026, 9, 20, 16, 2, tzinfo=UTC),
        verify_git=False,
    )
    assert first["inserted"] == 2
    assert second["unchanged"] == 2
    with connect(settings) as connection:
        predictions = [dict(row) for row in connection.execute("SELECT * FROM model_predictions")]
    assert len(predictions) == 2
    assert {row["decision"] for row in predictions} == {"PASS"}
    assert {row["eligibility_status"] for row in predictions} == {"ELIGIBLE_CLOSE_PROXY"}
    for row in predictions:
        assert row["model_win_probability"] + row["model_push_probability"] + row[
            "model_loss_probability"
        ] == pytest.approx(1.0)
