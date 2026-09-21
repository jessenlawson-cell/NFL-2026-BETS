from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pandas as pd

from nfl_bets.schemas import ARTIFACT_SCHEMAS
from scripts.rank_bets import audit_repository


def test_audit_reports_every_gate_blocking_bet_ranking(tmp_path) -> None:
    metadata = {
        "source": "fixture",
        "retrieved_at_utc": "2026-09-14T23:46:08Z",
        "source_updated_at_utc": None,
        "schema_version": "1.1.0",
        "content_hash": "fixture",
    }
    for name, schema in ARTIFACT_SCHEMAS.items():
        pd.DataFrame(columns=schema.columns).to_csv(tmp_path / f"{name}.csv", index=False)

    game = {
        **{column: None for column in ARTIFACT_SCHEMAS["games"].columns},
        **metadata,
        "game_id": "2026_02_BUF_NYJ",
        "season": 2026,
        "week": 2,
        "game_type": "REG",
        "kickoff_utc": "2026-09-20T17:00:00Z",
        "away_team": "BUF",
        "home_team": "NYJ",
    }
    pd.DataFrame([game]).to_csv(tmp_path / "games.csv", index=False)

    odds = []
    for market, selections, line in (
        ("spreads", ("HOME", "AWAY"), 3.0),
        ("totals", ("OVER", "UNDER"), 44.0),
    ):
        for selection in selections:
            odds.append(
                {
                    **{column: None for column in ARTIFACT_SCHEMAS["market_odds"].columns},
                    **metadata,
                    "snapshot_id": "close",
                    "provider_event_id": "event",
                    "game_id": game["game_id"],
                    "commence_time_utc": game["kickoff_utc"],
                    "bookmaker_key": "pinnacle",
                    "bookmaker_title": "Pinnacle",
                    "bookmaker_group": "sharp",
                    "market": market,
                    "selection": selection,
                    "point": line,
                    "canonical_line": line,
                    "american_price": -110,
                    "decimal_price": 1.9090909090909092,
                    "implied_probability": 0.5238095238095238,
                    "vig_free_probability": 0.5,
                    "overround": 1.0476190476190477,
                    "last_update_utc": "2026-09-14T23:46:05Z",
                    "source_updated_at_utc": "2026-09-14T23:46:05Z",
                }
            )
    pd.DataFrame(odds).to_csv(tmp_path / "market_odds.csv", index=False)

    (tmp_path / "PROJECT_STATE.md").write_text(
        "# State\n"
        "- **Current Season:** 2026\n"
        "- **Current Week:** 1\n"
        "- **Last Model Version:** 1.1.2 (LOCKED_UNTESTED_2026)\n"
        "- **Betting Status:** PASS-only\n",
        encoding="utf-8",
    )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "prospective_policy_1.1.2.json").write_text(
        json.dumps({"allowed_decisions": ["PASS"]}), encoding="utf-8"
    )
    runtime = tmp_path / "data" / "runtime"
    runtime.mkdir(parents=True)
    with sqlite3.connect(runtime / "nfl_bets.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE raw_snapshots ("
            "snapshot_id TEXT, snapshot_purpose TEXT, retrieved_at_utc TEXT)"
        )
        connection.execute(
            "INSERT INTO raw_snapshots VALUES (?,?,?)",
            ("close", "CLOSE", metadata["retrieved_at_utc"]),
        )

    report = audit_repository(tmp_path, datetime(2026, 9, 20, 15, tzinfo=UTC))

    assert report["games_evaluated"] == 1
    assert report["markets_evaluated"] == 2
    assert report["qualifying_count"] == 0
    assert set(report["blocking_reasons"]) >= {
        "PROJECT_STATE betting status is PASS-only",
        "prospective policy permits only PASS decisions",
        "latest market snapshot is CLOSE, not DECISION",
        "latest market quote is older than 30 minutes",
        "PROJECT_STATE current week 1 does not match upcoming market week 2",
        "no model predictions exist for the latest market snapshot",
    }
