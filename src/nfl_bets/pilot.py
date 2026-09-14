from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import SCHEDULED_SLOTS, _week_bucket, snapshot_odds
from nfl_bets.prospective import DEFAULT_MODEL_VERSION, predict_snapshot
from nfl_bets.util import atomic_write_text, sha256_bytes


def capture_pilot_slot(
    slot: str,
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Perform one user-initiated capture and immediately persist PASS-only predictions."""
    if slot not in SCHEDULED_SLOTS:
        raise ValueError(f"Pilot slot must be one of: {', '.join(sorted(SCHEDULED_SLOTS))}")
    resolved = settings or get_settings()
    odds = snapshot_odds(slot, settings=resolved)
    prediction = predict_snapshot(str(odds["snapshot_id"]), version=version, settings=resolved)
    return {
        "slot": slot,
        "odds": odds,
        "prediction": prediction,
        "decision": "PASS",
        "automatic_retry": False,
    }


def _snapshot_is_intact(row: dict[str, Any]) -> bool:
    path = Path(str(row["path"]))
    headers_path = Path(str(row["headers_path"])) if row.get("headers_path") else None
    return (
        path.exists()
        and sha256_bytes(path.read_bytes()) == row["content_hash"]
        and (headers_path is None or headers_path.exists())
    )


def pilot_status(
    week_bucket: str | None = None,
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    bucket = week_bucket or _week_bucket(datetime.now(UTC), resolved)
    slots: dict[str, Any] = {}
    with connect(resolved) as connection:
        failed_attempts = int(
            connection.execute(
                "SELECT COUNT(*) FROM api_requests WHERE week_bucket=? AND status!='COMPLETE'",
                (bucket,),
            ).fetchone()[0]
        )
        for slot in sorted(SCHEDULED_SLOTS):
            request = connection.execute(
                "SELECT * FROM api_requests WHERE week_bucket=? AND slot=? AND status='COMPLETE' "
                "ORDER BY completed_at_utc DESC LIMIT 1",
                (bucket, slot),
            ).fetchone()
            if request is None:
                slots[slot] = {"status": "MISSING"}
                continue
            snapshot = connection.execute(
                "SELECT * FROM raw_snapshots WHERE snapshot_id=?",
                (request["raw_snapshot_id"],),
            ).fetchone()
            quote_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM market_odds WHERE snapshot_id=?",
                    (request["raw_snapshot_id"],),
                ).fetchone()[0]
            )
            prediction_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM model_predictions "
                    "WHERE snapshot_id=? AND model_version=?",
                    (request["raw_snapshot_id"], version),
                ).fetchone()[0]
            )
            non_pass = int(
                connection.execute(
                    "SELECT COUNT(*) FROM model_predictions "
                    "WHERE snapshot_id=? AND decision!='PASS'",
                    (request["raw_snapshot_id"],),
                ).fetchone()[0]
            )
            raw_intact = snapshot is not None and _snapshot_is_intact(dict(snapshot))
            passed = raw_intact and quote_count > 0 and prediction_count > 0 and non_pass == 0
            slots[slot] = {
                "status": "PASSED" if passed else "FAILED",
                "request_id": request["request_id"],
                "snapshot_id": request["raw_snapshot_id"],
                "raw_intact": raw_intact,
                "quotes": quote_count,
                "predictions": prediction_count,
                "non_pass_decisions": non_pass,
            }
    passed_slots = sum(1 for value in slots.values() if value["status"] == "PASSED")
    status = "PASSED" if passed_slots == len(SCHEDULED_SLOTS) else "INCOMPLETE"
    report = {
        "pilot_type": "ONE_FULL_WEEK_MANUAL",
        "week_bucket": bucket,
        "model_version": version,
        "status": status,
        "required_slots": len(SCHEDULED_SLOTS),
        "passed_slots": passed_slots,
        "failed_attempts": failed_attempts,
        "automatic_retry": False,
        "decision_policy": "PASS-only",
        "slots": slots,
    }
    report_path = resolved.reports_dir / f"manual_pilot_{bucket}.json"
    atomic_write_text(report_path, json.dumps(report, sort_keys=True, indent=2))
    return {**report, "report": str(report_path)}
