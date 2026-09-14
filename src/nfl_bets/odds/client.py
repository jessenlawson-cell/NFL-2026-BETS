from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, immediate_transaction, initialize_database, transaction
from nfl_bets.odds.consensus import RETAIL_BOOKS, TEAM_NAME_TO_ID, ParsedBoard, parse_board
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_bytes, atomic_write_text, iso_utc, sha256_bytes

SCHEDULED_SLOTS = {
    "sunday_open_2000",
    "sunday_open_2330",
    "wednesday_0900",
    "wednesday_1700",
    "thursday_1200",
    "thursday_1930",
    "friday_1700",
    "saturday_0900",
    "saturday_1700",
    "saturday_2300",
    "sunday_0845",
    "sunday_1245",
    "sunday_1545",
    "sunday_1945",
    "monday_1200",
    "monday_1945",
}
SNAPSHOT_PURPOSES = {"DECISION", "CLOSE", "DIAGNOSTIC"}
SCHEDULED_SLOT_PURPOSES = {
    "sunday_open_2000": "DECISION",
    "sunday_open_2330": "DECISION",
    "wednesday_0900": "DECISION",
    "wednesday_1700": "DECISION",
    "thursday_1200": "DECISION",
    "thursday_1930": "CLOSE",
    "friday_1700": "DECISION",
    "saturday_0900": "DECISION",
    "saturday_1700": "DECISION",
    "saturday_2300": "DECISION",
    "sunday_0845": "DECISION",
    "sunday_1245": "CLOSE",
    "sunday_1545": "CLOSE",
    "sunday_1945": "CLOSE",
    "monday_1200": "DECISION",
    "monday_1945": "CLOSE",
}
REQUEST_CREDIT_COST = 2
REQUEST_KIND = "full-board"
RECONCILIATION_RESOLUTIONS = {
    "SAFE_TO_RETRY_NO_CALL",
    "PROVIDER_CONFIRMED_NOT_BILLED",
    "PROVIDER_CONFIRMED_BILLED",
}
SAFE_RESPONSE_HEADERS = {
    "content-length",
    "content-type",
    "date",
    "request-id",
    "x-request-id",
    "x-requests-last",
    "x-requests-remaining",
    "x-requests-used",
}


class QuotaError(RuntimeError):
    pass


class DuplicateCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class RequestReservation:
    request_id: str
    week_bucket: str
    idempotency_key: str | None
    attempt_number: int | None


def _redact(message: str, secret: str) -> str:
    return message.replace(secret, "***") if secret else message


def _week_bucket(now: datetime, settings: Settings) -> str:
    local = now.astimezone(settings.tz)
    days_since_tuesday = (local.weekday() - 1) % 7
    start = (local - timedelta(days=days_since_tuesday)).date()
    return start.isoformat()


def _snapshot_purpose(slot: str, purpose: str | None) -> str:
    normalized = (purpose or SCHEDULED_SLOT_PURPOSES.get(slot, "DIAGNOSTIC")).strip().upper()
    if normalized not in SNAPSHOT_PURPOSES:
        raise ValueError(f"Snapshot purpose must be one of: {', '.join(sorted(SNAPSHOT_PURPOSES))}")
    configured = SCHEDULED_SLOT_PURPOSES.get(slot)
    if configured is not None and normalized != configured:
        raise ValueError(f"Scheduled slot {slot} is registered as {configured}, not {normalized}")
    return normalized


def _request_kind(purpose: str) -> str:
    return f"{REQUEST_KIND}:{purpose.lower()}"


def scheduled_capture_key(
    week_bucket: str, slot: str, request_kind: str | None = None
) -> str:
    if slot not in SCHEDULED_SLOTS:
        raise ValueError(f"Stable capture keys require a configured slot, not {slot!r}")
    resolved_kind = request_kind or _request_kind(SCHEDULED_SLOT_PURPOSES[slot])
    return f"scheduled-capture:v1:{week_bucket}:{slot}:{resolved_kind}"


def _apply_quota_guard(connection: Any, week: str, settings: Settings, scheduled: bool) -> None:
    total = connection.execute(
        "SELECT COUNT(*) FROM api_requests WHERE week_bucket=?", (week,)
    ).fetchone()[0]
    scheduled_count = connection.execute(
        "SELECT COUNT(*) FROM api_requests WHERE week_bucket=? AND slot!='manual'",
        (week,),
    ).fetchone()[0]
    latest_provider_remaining = connection.execute(
        "SELECT provider_requests_remaining FROM api_requests "
        "WHERE provider_requests_remaining IS NOT NULL "
        "AND substr(started_at_utc,1,7)=? "
        "ORDER BY completed_at_utc DESC LIMIT 1",
        (iso_utc()[:7],),
    ).fetchone()
    if (
        latest_provider_remaining is not None
        and int(latest_provider_remaining[0]) < REQUEST_CREDIT_COST
    ):
        raise QuotaError(
            "Provider reports fewer than two credits; consolidated capture was not attempted"
        )
    if total >= settings.weekly_call_limit:
        raise QuotaError(f"Weekly request ceiling reached for bucket {week}")
    if scheduled and scheduled_count >= settings.weekly_call_limit - settings.manual_call_reserve:
        raise QuotaError("Scheduled request reserve exhausted; four calls remain manual-only")


def _reserve_request(
    slot: str, purpose: str, settings: Settings, now: datetime
) -> RequestReservation:
    week = _week_bucket(now, settings)
    scheduled = slot in SCHEDULED_SLOTS
    request_kind = _request_kind(purpose)
    key = scheduled_capture_key(week, slot, request_kind) if scheduled else None
    normalized_slot = slot if scheduled else "manual"
    duplicate_status: str | None = None
    reservation: RequestReservation | None = None
    with immediate_transaction(settings) as connection:
        attempt_number: int | None = None
        existing = None
        if key is not None:
            existing = connection.execute(
                "SELECT * FROM capture_reservations WHERE idempotency_key=? OR "
                "(week_bucket=? AND slot=? AND request_kind=?) "
                "ORDER BY (idempotency_key=?) DESC LIMIT 1",
                (key, week, slot, REQUEST_KIND, key),
            ).fetchone()
            legacy_key = existing is not None and existing["idempotency_key"] != key
            if existing is not None and (legacy_key or existing["status"] != "RETRY_AUTHORIZED"):
                duplicate_status = (
                    f"LEGACY_{existing['status']}" if legacy_key else str(existing["status"])
                )
                connection.execute(
                    "UPDATE capture_reservations SET updated_at_utc=?,"
                    "reservation_conflicts=reservation_conflicts+1 WHERE idempotency_key=?",
                    (iso_utc(now), existing["idempotency_key"]),
                )
            elif existing is not None:
                attempt_number = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(attempt_number),0)+1 FROM api_requests "
                        "WHERE idempotency_key=?",
                        (key,),
                    ).fetchone()[0]
                )
            else:
                attempt_number = 1
        if duplicate_status is None:
            _apply_quota_guard(connection, week, settings, scheduled)
            request_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO api_requests("
                "request_id,idempotency_key,attempt_number,slot,request_kind,week_bucket,"
                "started_at_utc,status,reconciliation_status) "
                "VALUES (?,?,?,?,?,?,?,'RESERVED','NOT_REQUIRED')",
                (
                    request_id,
                    key,
                    attempt_number,
                    normalized_slot,
                    request_kind,
                    week,
                    iso_utc(now),
                ),
            )
            if key is not None:
                if existing is None:
                    connection.execute(
                        "INSERT INTO capture_reservations("
                        "idempotency_key,week_bucket,slot,request_kind,created_at_utc,"
                        "updated_at_utc,status,active_request_id) "
                        "VALUES (?,?,?,?,?,?,'RESERVED',?)",
                        (key, week, slot, request_kind, iso_utc(now), iso_utc(now), request_id),
                    )
                else:
                    connection.execute(
                        "UPDATE capture_reservations SET updated_at_utc=?,status='RESERVED',"
                        "active_request_id=?,reconciliation_status='RETRY_CONSUMED' "
                        "WHERE idempotency_key=? AND status='RETRY_AUTHORIZED'",
                        (iso_utc(now), request_id, key),
                    )
            reservation = RequestReservation(request_id, week, key, attempt_number)
    if duplicate_status is not None:
        raise DuplicateCaptureError(
            f"Scheduled capture is already reserved with status {duplicate_status}; "
            "no provider call was made"
        )
    if reservation is None:
        raise RuntimeError("Capture reservation was not created")
    return reservation


def _safe_int_header(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _sanitized_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Persist only operational headers; cookies, authorization, and key echoes are dropped."""
    return {
        key.lower(): value for key, value in headers.items() if key.lower() in SAFE_RESPONSE_HEADERS
    }


def _match_games(payload: list[dict[str, Any]], settings: Settings) -> dict[str, str | None]:
    with connect(settings) as connection:
        games = connection.execute(
            "SELECT game_id,kickoff_utc,home_team,away_team FROM games"
        ).fetchall()
    matches: dict[str, str | None] = {}
    used_game_ids: set[str] = set()
    tolerance = timedelta(hours=settings.schedule_match_tolerance_hours)
    for event in payload:
        event_id = str(event["id"])
        home_id = TEAM_NAME_TO_ID.get(str(event["home_team"]))
        away_id = TEAM_NAME_TO_ID.get(str(event["away_team"]))
        commence = datetime.fromisoformat(str(event["commence_time"]).replace("Z", "+00:00"))
        candidates = []
        for game in games:
            kickoff = datetime.fromisoformat(str(game["kickoff_utc"]).replace("Z", "+00:00"))
            if (
                game["home_team"] == home_id
                and game["away_team"] == away_id
                and abs(kickoff - commence) <= tolerance
            ):
                candidates.append(str(game["game_id"]))
        if len(candidates) > 1:
            raise ValueError(f"Provider event {event_id} maps to multiple nflverse games")
        match = candidates[0] if candidates else None
        if match is not None and match in used_game_ids:
            raise ValueError(f"Multiple provider events map to nflverse game {match}")
        if match is not None:
            used_game_ids.add(match)
        matches[event_id] = match
    return matches


def _persist_parsed(parsed: ParsedBoard, settings: Settings) -> None:
    quote_columns = ARTIFACT_SCHEMAS["market_odds"].columns
    with transaction(settings) as connection:
        if parsed.quotes:
            placeholders = ",".join("?" for _ in quote_columns)
            connection.executemany(
                f"INSERT INTO market_odds ({','.join(quote_columns)}) VALUES ({placeholders})",
                [tuple(row[column] for column in quote_columns) for row in parsed.quotes],
            )
        if parsed.consensus:
            columns = tuple(parsed.consensus[0])
            placeholders = ",".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO market_consensus ({','.join(columns)}) VALUES ({placeholders})",
                [tuple(row[column] for column in columns) for row in parsed.consensus],
            )
    with connect(settings) as connection:
        rows = [
            dict(row)
            for row in connection.execute("SELECT * FROM market_odds ORDER BY retrieved_at_utc")
        ]
    frame = (
        pl.DataFrame(rows).select(quote_columns)
        if rows
        else pl.DataFrame({c: [] for c in quote_columns})
    )
    atomic_write_text(settings.root / "market_odds.csv", frame.write_csv())


def _update_reservation(
    connection: Any,
    reservation: RequestReservation,
    *,
    status: str,
    reconciliation_status: str | None = None,
    increment_provider_calls: bool = False,
) -> None:
    if reservation.idempotency_key is None:
        return
    assignments = ["updated_at_utc=?", "status=?"]
    values: list[object] = [iso_utc(), status]
    if reconciliation_status is not None:
        assignments.append("reconciliation_status=?")
        values.append(reconciliation_status)
    if increment_provider_calls:
        assignments.append("provider_call_count=provider_call_count+1")
    values.append(reservation.idempotency_key)
    connection.execute(
        f"UPDATE capture_reservations SET {','.join(assignments)} WHERE idempotency_key=?",
        tuple(values),
    )


def _mark_pre_call_failure(
    reservation: RequestReservation,
    settings: Settings,
    message: str,
) -> None:
    with immediate_transaction(settings) as connection:
        connection.execute(
            "UPDATE api_requests SET completed_at_utc=?,status='PRE_CALL_FAILED',"
            "reconciliation_status='SAFE_RETRY_REQUIRES_OPERATOR',error_message=? "
            "WHERE request_id=? AND provider_call_started_at_utc IS NULL",
            (iso_utc(), message[:2000], reservation.request_id),
        )
        _update_reservation(
            connection,
            reservation,
            status="RECONCILIATION_REQUIRED",
            reconciliation_status="SAFE_RETRY_REQUIRES_OPERATOR",
        )


def _mark_provider_call_started(
    reservation: RequestReservation,
    settings: Settings,
) -> None:
    started = iso_utc()
    with immediate_transaction(settings) as connection:
        updated = connection.execute(
            "UPDATE api_requests SET provider_call_started_at_utc=?,status='CALL_STARTED' "
            "WHERE request_id=? AND status='RESERVED' AND provider_call_started_at_utc IS NULL",
            (started, reservation.request_id),
        )
        if updated.rowcount != 1:
            raise RuntimeError("Capture reservation could not enter the provider-call state")
        _update_reservation(
            connection,
            reservation,
            status="IN_FLIGHT",
            increment_provider_calls=True,
        )


def _mark_ambiguous_failure(
    reservation: RequestReservation,
    settings: Settings,
    message: str,
) -> None:
    with immediate_transaction(settings) as connection:
        connection.execute(
            "UPDATE api_requests SET completed_at_utc=?,status='AMBIGUOUS_FAILURE',"
            "reconciliation_status='PROVIDER_CHECK_REQUIRED',error_message=? "
            "WHERE request_id=?",
            (iso_utc(), message[:2000], reservation.request_id),
        )
        _update_reservation(
            connection,
            reservation,
            status="RECONCILIATION_REQUIRED",
            reconciliation_status="PROVIDER_CHECK_REQUIRED",
        )


def _mark_response_received(
    reservation: RequestReservation,
    settings: Settings,
    response: httpx.Response,
) -> tuple[str, dict[str, str], str | None]:
    received = iso_utc()
    safe_headers = _sanitized_response_headers(response.headers)
    provider_request_id = safe_headers.get("x-request-id") or safe_headers.get("request-id")
    with immediate_transaction(settings) as connection:
        connection.execute(
            "UPDATE api_requests SET provider_response_received_at_utc=?,"
            "status='RESPONSE_RECEIVED',"
            "http_status=?,provider_request_id=?,provider_requests_used=?,"
            "provider_requests_remaining=?,provider_requests_last=? WHERE request_id=?",
            (
                received,
                response.status_code,
                provider_request_id,
                _safe_int_header(response.headers, "x-requests-used"),
                _safe_int_header(response.headers, "x-requests-remaining"),
                _safe_int_header(response.headers, "x-requests-last"),
                reservation.request_id,
            ),
        )
        _update_reservation(
            connection,
            reservation,
            status="RESPONSE_RECEIVED",
            reconciliation_status="NO_RETRY_RESPONSE_RECEIVED",
        )
    return received, safe_headers, provider_request_id


def _mark_raw_persist_failure(
    reservation: RequestReservation,
    settings: Settings,
    message: str,
) -> None:
    with immediate_transaction(settings) as connection:
        connection.execute(
            "UPDATE api_requests SET completed_at_utc=?,status='RAW_PERSIST_FAILED',"
            "reconciliation_status='NO_RETRY_RESPONSE_RECEIVED',error_message=? "
            "WHERE request_id=?",
            (iso_utc(), message[:2000], reservation.request_id),
        )
        _update_reservation(
            connection,
            reservation,
            status="RECONCILIATION_REQUIRED",
            reconciliation_status="NO_RETRY_RESPONSE_RECEIVED",
        )


def capture_reconciliation_status(
    slot: str,
    week_bucket: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    key = scheduled_capture_key(week_bucket, slot)
    with connect(resolved) as connection:
        row = connection.execute(
            "SELECT r.*,a.status AS attempt_status,a.attempt_number,"
            "a.provider_call_started_at_utc,a.provider_response_received_at_utc,"
            "a.provider_request_id,a.raw_snapshot_id "
            "FROM capture_reservations r LEFT JOIN api_requests a "
            "ON a.request_id=r.active_request_id WHERE r.idempotency_key=? OR "
            "(r.week_bucket=? AND r.slot=? AND r.request_kind=?) "
            "ORDER BY (r.idempotency_key=?) DESC LIMIT 1",
            (key, week_bucket, slot, REQUEST_KIND, key),
        ).fetchone()
    if row is None:
        return {
            "status": "UNRESERVED",
            "idempotency_key": key,
            "week_bucket": week_bucket,
            "slot": slot,
            "safe_to_call": True,
        }
    result = dict(row)
    result["legacy_key"] = result["idempotency_key"] != key
    result["provider_request_id_recorded"] = bool(result.pop("provider_request_id", None))
    with connect(resolved) as connection:
        result["reconciliation_events"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM capture_reconciliations WHERE idempotency_key=?",
                (result["idempotency_key"],),
            ).fetchone()[0]
        )
    result["safe_to_call"] = result["status"] == "RETRY_AUTHORIZED" and not result["legacy_key"]
    result["automatic_retry"] = False
    return result


def reconcile_scheduled_capture(
    slot: str,
    week_bucket: str,
    resolution: str,
    note: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Record a human/provider reconciliation; never contacts or retries the provider."""
    resolved = settings or get_settings()
    initialize_database(resolved)
    normalized = resolution.strip().upper()
    if normalized not in RECONCILIATION_RESOLUTIONS:
        raise ValueError(
            f"Resolution must be one of: {', '.join(sorted(RECONCILIATION_RESOLUTIONS))}"
        )
    if not note.strip():
        raise ValueError("A non-empty reconciliation note is required")
    key = scheduled_capture_key(week_bucket, slot)
    reconciled_at = iso_utc()
    with immediate_transaction(resolved) as connection:
        row = connection.execute(
            "SELECT r.idempotency_key,r.status AS reservation_status,"
            "r.reconciliation_status AS current_reconciliation_status,r.active_request_id,"
            "a.status AS attempt_status,a.provider_call_started_at_utc,"
            "a.provider_response_received_at_utc,a.raw_snapshot_id "
            "FROM capture_reservations r JOIN api_requests a "
            "ON a.request_id=r.active_request_id WHERE r.idempotency_key=? OR "
            "(r.week_bucket=? AND r.slot=? AND r.request_kind=?) "
            "ORDER BY (r.idempotency_key=?) DESC LIMIT 1",
            (key, week_bucket, slot, REQUEST_KIND, key),
        ).fetchone()
        if row is None:
            raise ValueError("No scheduled capture reservation exists for this key")
        call_started = row["provider_call_started_at_utc"] is not None
        response_received = row["provider_response_received_at_utc"] is not None
        raw_saved = row["raw_snapshot_id"] is not None
        legacy_provider_check = (
            row["current_reconciliation_status"] == "LEGACY_PROVIDER_CHECK_REQUIRED"
        )
        open_reconciliation = row["reservation_status"] in {
            "RESERVED",
            "IN_FLIGHT",
            "RESPONSE_RECEIVED",
            "RECONCILIATION_REQUIRED",
        }
        if not open_reconciliation:
            raise ValueError(
                f"Reservation status {row['reservation_status']} is not open for reconciliation"
            )
        if normalized == "SAFE_TO_RETRY_NO_CALL":
            if call_started or response_received or raw_saved:
                raise ValueError("A provider call may have started; no-call retry is unsafe")
            reservation_status = "RETRY_AUTHORIZED"
            attempt_status = "ABANDONED_NO_CALL"
        elif normalized == "PROVIDER_CONFIRMED_NOT_BILLED":
            if (not call_started and not legacy_provider_check) or response_received or raw_saved:
                raise ValueError(
                    "Not-billed reconciliation requires a started call with no response or raw body"
                )
            reservation_status = "RETRY_AUTHORIZED"
            attempt_status = "RECONCILED_NOT_BILLED"
        else:
            if not call_started and not legacy_provider_check:
                raise ValueError("Billed reconciliation requires a recorded provider-call start")
            reservation_status = "CLOSED_NO_RETRY"
            attempt_status = "RECONCILED_BILLED"
        connection.execute(
            "INSERT INTO capture_reconciliations("
            "reconciliation_id,idempotency_key,request_id,recorded_at_utc,resolution,note,source) "
            "VALUES (?,?,?,?,?,?,'operator')",
            (
                str(uuid.uuid4()),
                row["idempotency_key"],
                row["active_request_id"],
                reconciled_at,
                normalized,
                note.strip(),
            ),
        )
        connection.execute(
            "UPDATE api_requests SET completed_at_utc=COALESCE(completed_at_utc,?),status=?,"
            "reconciliation_status=? WHERE request_id=?",
            (reconciled_at, attempt_status, normalized, row["active_request_id"]),
        )
        connection.execute(
            "UPDATE capture_reservations SET updated_at_utc=?,status=?,"
            "reconciliation_status=?,reconciled_at_utc=?,reconciliation_note=? "
            "WHERE idempotency_key=?",
            (
                reconciled_at,
                reservation_status,
                normalized,
                reconciled_at,
                note.strip(),
                row["idempotency_key"],
            ),
        )
    return capture_reconciliation_status(slot, week_bucket, resolved)


def snapshot_odds(
    slot: str,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
    *,
    purpose: str | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    configured_key = resolved.the_odds_api_key
    api_key = configured_key.get_secret_value() if configured_key else None
    api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is not set")
    snapshot_purpose = _snapshot_purpose(slot, purpose)
    now = datetime.now(UTC)
    reservation = _reserve_request(slot, snapshot_purpose, resolved, now)
    params = {
        "apiKey": api_key,
        "bookmakers": ",".join(("pinnacle", *RETAIL_BOOKS)),
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = f"{resolved.odds_base_url}/v4/sports/{resolved.odds_sport_key}/odds"
    owns_client = client is None
    try:
        active_client = client or httpx.Client(timeout=resolved.odds_timeout_seconds)
    except BaseException as exc:
        _mark_pre_call_failure(reservation, resolved, type(exc).__name__)
        raise RuntimeError(
            "Odds client failed before the provider call; operator reconciliation is required"
        ) from None
    try:
        _mark_provider_call_started(reservation, resolved)
        response = active_client.get(url, params=params)
        retrieved_at, safe_headers, provider_request_id = _mark_response_received(
            reservation, resolved, response
        )
    except httpx.RequestError as exc:
        _mark_ambiguous_failure(reservation, resolved, _redact(str(exc), api_key))
        raise RuntimeError("Odds request failed ambiguously and was not retried") from None
    finally:
        if owns_client:
            active_client.close()

    snapshot_id = str(uuid.uuid4())
    date_path = now.strftime("%Y/%m")
    file_stem = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{snapshot_id}"
    raw_path = resolved.raw_dir / "odds" / date_path / f"{file_stem}.json"
    headers_path = resolved.raw_dir / "odds" / date_path / f"{file_stem}.headers.json"
    try:
        atomic_write_bytes(raw_path, response.content)
        atomic_write_text(headers_path, json.dumps(safe_headers, sort_keys=True, indent=2))
    except BaseException as exc:
        _mark_raw_persist_failure(reservation, resolved, type(exc).__name__)
        raise RuntimeError(
            "Provider responded but raw evidence could not be saved; do not retry the call"
        ) from None
    content_hash = sha256_bytes(response.content)
    with immediate_transaction(resolved) as connection:
        connection.execute(
            "INSERT INTO raw_snapshots("
            "snapshot_id,provider,kind,snapshot_purpose,path,headers_path,retrieved_at_utc,"
            "content_hash,byte_count) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                snapshot_id,
                "the-odds-api",
                "full-board",
                snapshot_purpose,
                str(raw_path),
                str(headers_path),
                retrieved_at,
                content_hash,
                len(response.content),
            ),
        )
        connection.execute(
            "UPDATE api_requests SET completed_at_utc=?,status='RAW_SAVED',"
            "raw_snapshot_id=? WHERE request_id=?",
            (
                retrieved_at,
                snapshot_id,
                reservation.request_id,
            ),
        )
        _update_reservation(
            connection,
            reservation,
            status="RAW_SAVED",
            reconciliation_status="NO_RETRY_RAW_SAVED",
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        with immediate_transaction(resolved) as connection:
            connection.execute(
                "UPDATE api_requests SET status='HTTP_FAILED',"
                "reconciliation_status='NO_RETRY_RAW_SAVED',error_message=? WHERE request_id=?",
                (_redact(str(exc), api_key)[:2000], reservation.request_id),
            )
            _update_reservation(
                connection,
                reservation,
                status="CLOSED_NO_RETRY",
                reconciliation_status="NO_RETRY_RAW_SAVED",
            )
        raise RuntimeError(
            f"Odds provider returned HTTP {response.status_code}; raw response was saved"
        ) from None
    try:
        payload = json.loads(response.content)
        if not isinstance(payload, list):
            raise ValueError("Odds response is not a board array")
        matches = _match_games(payload, resolved)
        parsed = parse_board(
            payload,
            snapshot_id,
            retrieved_at,
            resolved.schema_version,
            matches,
            resolved.odds_freshness_minutes,
        )
        unmatched = [event_id for event_id, game_id in matches.items() if game_id is None]
        if unmatched:
            raise ValueError(
                f"One-to-one nflverse game mapping failed for provider events: {unmatched}"
            )
        _persist_parsed(parsed, resolved)
    except BaseException as exc:
        with immediate_transaction(resolved) as connection:
            connection.execute(
                "UPDATE api_requests SET status='PARSE_FAILED',"
                "reconciliation_status='NO_RETRY_RAW_SAVED',error_message=? WHERE request_id=?",
                (str(exc)[:2000], reservation.request_id),
            )
            _update_reservation(
                connection,
                reservation,
                status="CLOSED_NO_RETRY",
                reconciliation_status="NO_RETRY_RAW_SAVED",
            )
        raise
    with immediate_transaction(resolved) as connection:
        connection.execute(
            "UPDATE api_requests SET status='COMPLETE',reconciliation_status='NOT_REQUIRED' "
            "WHERE request_id=?",
            (reservation.request_id,),
        )
        _update_reservation(
            connection,
            reservation,
            status="COMPLETE",
            reconciliation_status="NOT_REQUIRED",
        )
    return {
        "request_id": reservation.request_id,
        "snapshot_id": snapshot_id,
        "snapshot_purpose": snapshot_purpose,
        "week_bucket": reservation.week_bucket,
        "idempotency_key": reservation.idempotency_key,
        "attempt_number": reservation.attempt_number,
        "events": len(payload),
        "quotes": len(parsed.quotes),
        "consensus_markets": len(parsed.consensus),
        "provider_credit_cost": _safe_int_header(response.headers, "x-requests-last"),
        "provider_request_id_recorded": provider_request_id is not None,
        "automatic_retry": False,
        "raw_path": raw_path,
    }
