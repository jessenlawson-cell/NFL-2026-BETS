from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import (
    SCHEDULED_SLOTS,
    _week_bucket,
    capture_reconciliation_status,
    reconcile_scheduled_capture,
    snapshot_odds,
)
from nfl_bets.prospective import DEFAULT_MODEL_VERSION, load_frozen_bundle, predict_snapshot
from nfl_bets.util import atomic_write_text, sha256_bytes

PREFLIGHT_REQUIRED_FILES = (
    "games.csv",
    "team_metrics.csv",
    "model_history.csv",
    "MODEL_SPEC_V1_1.md",
)
PREFLIGHT_REQUIRED_TABLES = {
    "api_requests",
    "capture_reconciliations",
    "capture_reservations",
    "games",
    "market_odds",
    "model_predictions",
    "prospective_evaluations",
}


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _runtime_paths_aligned(settings: Settings) -> bool:
    return settings.root == Path("/workspace")


def _check(name: str, passed: bool, detail: str, *, blocking: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else ("FAIL" if blocking else "WARNING"),
        "blocking": blocking,
        "detail": detail,
    }


def preflight_pilot(
    slot: str | None = None,
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Verify capture readiness without contacting the odds provider."""
    resolved = settings or get_settings()
    checks: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    bucket = _week_bucket(now, resolved)

    in_container = _running_in_container()
    checks.append(
        _check(
            "docker_runtime",
            in_container,
            "Running inside Docker." if in_container else "Run through Docker Compose.",
        )
    )
    root_aligned = _runtime_paths_aligned(resolved) if in_container else False
    checks.append(
        _check(
            "runtime_paths",
            root_aligned,
            f"Repository root is {resolved.root}.",
        )
    )

    key = resolved.the_odds_api_key
    key_present = bool(key and key.get_secret_value().strip())
    checks.append(
        _check(
            "api_key",
            key_present,
            "API key is present (value hidden)." if key_present else "THE_ODDS_API_KEY is missing.",
        )
    )

    missing_files = [
        name for name in PREFLIGHT_REQUIRED_FILES if not (resolved.root / name).is_file()
    ]
    checks.append(
        _check(
            "authoritative_files",
            not missing_files,
            "Required files are present."
            if not missing_files
            else f"Missing: {', '.join(missing_files)}",
        )
    )

    schedule_path = resolved.root / "config" / "scheduled_slots.toml"
    schedule_ok = False
    schedule_detail = "Schedule configuration is missing."
    try:
        schedule = tomllib.loads(schedule_path.read_text(encoding="utf-8"))
        configured_slots = set(schedule.get("slots", {}))
        schedule_ok = (
            configured_slots == SCHEDULED_SLOTS
            and schedule.get("timezone") == resolved.timezone
            and schedule.get("weekly_call_limit") == resolved.weekly_call_limit
            and schedule.get("manual_reserve") == resolved.manual_call_reserve
        )
        schedule_detail = (
            f"All {len(SCHEDULED_SLOTS)} slots match {resolved.timezone}."
            if schedule_ok
            else "Schedule settings do not match runtime safeguards."
        )
    except (OSError, tomllib.TOMLDecodeError):
        pass
    checks.append(_check("schedule", schedule_ok, schedule_detail))

    slot_ok = slot is None or slot in SCHEDULED_SLOTS
    checks.append(
        _check(
            "requested_slot",
            slot_ok,
            "No slot selected; general readiness only."
            if slot is None
            else (f"Configured slot: {slot}." if slot_ok else f"Unknown slot: {slot}."),
        )
    )

    database_ok = False
    database_detail = "Database could not be checked."
    quota_detail = "Quota could not be checked."
    quota_ok = False
    provider_detail = "No provider quota header has been recorded yet."
    provider_ok = True
    provider_blocking = False
    duplicate_ok = True
    duplicate_detail = "No completed capture exists for the selected slot."
    try:
        initialize_database(resolved)
        with connect(resolved) as connection:
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            database_ok = quick_check == "ok" and tables >= PREFLIGHT_REQUIRED_TABLES
            database_detail = (
                "SQLite integrity and required tables are valid."
                if database_ok
                else "SQLite integrity or required tables failed validation."
            )
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM api_requests WHERE week_bucket=?", (bucket,)
                ).fetchone()[0]
            )
            scheduled = int(
                connection.execute(
                    "SELECT COUNT(*) FROM api_requests WHERE week_bucket=? AND slot!='manual'",
                    (bucket,),
                ).fetchone()[0]
            )
            local_remaining = resolved.weekly_call_limit - total
            scheduled_remaining = (
                resolved.weekly_call_limit - resolved.manual_call_reserve - scheduled
            )
            quota_ok = local_remaining > 0 and scheduled_remaining > 0
            quota_detail = (
                f"Local guard: {local_remaining} total and {scheduled_remaining} "
                "scheduled requests remain."
            )
            provider_row = connection.execute(
                "SELECT provider_requests_remaining FROM api_requests "
                "WHERE provider_requests_remaining IS NOT NULL "
                "ORDER BY completed_at_utc DESC LIMIT 1"
            ).fetchone()
            if provider_row is not None:
                provider_remaining = int(provider_row[0])
                provider_ok = provider_remaining >= 2
                provider_blocking = True
                provider_detail = f"Last recorded provider balance: {provider_remaining} credits."
            if slot_ok and slot is not None:
                reservation = connection.execute(
                    "SELECT status,reconciliation_status FROM capture_reservations "
                    "WHERE week_bucket=? AND slot=? AND request_kind='full-board'",
                    (bucket, slot),
                ).fetchone()
                if reservation is None:
                    legacy = connection.execute(
                        "SELECT status FROM api_requests WHERE week_bucket=? AND slot=? LIMIT 1",
                        (bucket, slot),
                    ).fetchone()
                    duplicate_ok = legacy is None
                    duplicate_detail = (
                        "No reservation exists for this slot."
                        if duplicate_ok
                        else "A legacy capture exists for this slot; do not spend credits twice."
                    )
                else:
                    duplicate_ok = reservation["status"] == "RETRY_AUTHORIZED"
                    duplicate_detail = (
                        "An operator-authorized reconciled retry is ready."
                        if duplicate_ok
                        else (
                            f"Slot reservation is {reservation['status']} "
                            f"({reservation['reconciliation_status']}); do not call the provider."
                        )
                    )
    except Exception as exc:  # converted to a safe readiness result
        database_detail = f"Database check failed: {type(exc).__name__}."
    checks.extend(
        [
            _check("database", database_ok, database_detail),
            _check("local_quota", quota_ok, quota_detail),
            _check("provider_quota", provider_ok, provider_detail, blocking=provider_blocking),
            _check("duplicate_slot", duplicate_ok, duplicate_detail),
        ]
    )

    try:
        bundle = load_frozen_bundle(version=version, settings=resolved)
        model_ok = bundle.policy.get("status") == "LOCKED_UNTESTED_2026"
        model_detail = (
            f"Frozen model {version} passed artifact and source checks."
            if model_ok
            else f"Frozen model {version} has an invalid status."
        )
    except Exception as exc:  # never expose paths or secret-bearing exception text
        model_ok = False
        model_detail = f"Frozen model check failed: {type(exc).__name__}."
    checks.append(_check("frozen_model", model_ok, model_detail))

    ready = all(row["status"] != "FAIL" for row in checks)
    return {
        "status": "READY" if ready else "NOT_READY",
        "safe_to_capture": ready,
        "zero_credit_check": True,
        "provider_contacted": False,
        "checked_at_utc": now.isoformat().replace("+00:00", "Z"),
        "week_bucket": bucket,
        "slot": slot,
        "model_version": version,
        "decision_policy": "PASS-only",
        "checks": checks,
    }


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


def reconcile_pilot_slot(
    slot: str,
    week_bucket: str,
    resolution: str | None = None,
    note: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Inspect or explicitly reconcile one scheduled capture without provider contact."""
    resolved = settings or get_settings()
    if resolution is None:
        return capture_reconciliation_status(slot, week_bucket, resolved)
    return reconcile_scheduled_capture(
        slot,
        week_bucket,
        resolution,
        note or "",
        resolved,
    )


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
        reservation_summary = {
            str(row["status"]): int(row["count"])
            for row in connection.execute(
                "SELECT status,COUNT(*) AS count FROM capture_reservations "
                "WHERE week_bucket=? GROUP BY status ORDER BY status",
                (bucket,),
            )
        }
        reservation_conflicts = int(
            connection.execute(
                "SELECT COALESCE(SUM(reservation_conflicts),0) FROM capture_reservations "
                "WHERE week_bucket=?",
                (bucket,),
            ).fetchone()[0]
        )
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
        "reservation_statuses": reservation_summary,
        "reservation_conflicts": reservation_conflicts,
        "automatic_retry": False,
        "decision_policy": "PASS-only",
        "slots": slots,
    }
    report_path = resolved.reports_dir / f"manual_pilot_{bucket}.json"
    atomic_write_text(report_path, json.dumps(report, sort_keys=True, indent=2))
    return {**report, "report": str(report_path)}
