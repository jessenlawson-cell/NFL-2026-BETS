from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database, transaction
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
REQUEST_CREDIT_COST = 2
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


def _redact(message: str, secret: str) -> str:
    return message.replace(secret, "***") if secret else message


def _week_bucket(now: datetime, settings: Settings) -> str:
    local = now.astimezone(settings.tz)
    days_since_tuesday = (local.weekday() - 1) % 7
    start = (local - timedelta(days=days_since_tuesday)).date()
    return start.isoformat()


def _reserve_request(slot: str, settings: Settings, now: datetime) -> tuple[str, str]:
    week = _week_bucket(now, settings)
    scheduled = slot in SCHEDULED_SLOTS
    with transaction(settings) as connection:
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
            (iso_utc(now)[:7],),
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
        if (
            scheduled
            and scheduled_count >= settings.weekly_call_limit - settings.manual_call_reserve
        ):
            raise QuotaError("Scheduled request reserve exhausted; four calls remain manual-only")
        request_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO api_requests("
            "request_id,slot,request_kind,week_bucket,started_at_utc,status) "
            "VALUES (?,?,?,?,?,'STARTED')",
            (request_id, slot if scheduled else "manual", "full-board", week, iso_utc(now)),
        )
    return request_id, week


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


def snapshot_odds(
    slot: str,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    configured_key = resolved.the_odds_api_key
    api_key = configured_key.get_secret_value() if configured_key else None
    api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is not set")
    now = datetime.now(UTC)
    request_id, week = _reserve_request(slot, resolved, now)
    params = {
        "apiKey": api_key,
        "bookmakers": ",".join(("pinnacle", *RETAIL_BOOKS)),
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = f"{resolved.odds_base_url}/v4/sports/{resolved.odds_sport_key}/odds"
    owns_client = client is None
    active_client = client or httpx.Client(timeout=resolved.odds_timeout_seconds)
    try:
        response = active_client.get(url, params=params)
    except httpx.RequestError as exc:
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE api_requests SET completed_at_utc=?,status='AMBIGUOUS_FAILURE',"
                "error_message=? "
                "WHERE request_id=?",
                (iso_utc(), _redact(str(exc), api_key)[:2000], request_id),
            )
        raise RuntimeError("Odds request failed ambiguously and was not retried") from None
    finally:
        if owns_client:
            active_client.close()

    retrieved_at = iso_utc()
    snapshot_id = str(uuid.uuid4())
    date_path = now.strftime("%Y/%m")
    file_stem = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{snapshot_id}"
    raw_path = resolved.raw_dir / "odds" / date_path / f"{file_stem}.json"
    headers_path = resolved.raw_dir / "odds" / date_path / f"{file_stem}.headers.json"
    atomic_write_bytes(raw_path, response.content)
    safe_headers = _sanitized_response_headers(response.headers)
    atomic_write_text(headers_path, json.dumps(safe_headers, sort_keys=True, indent=2))
    content_hash = sha256_bytes(response.content)
    with transaction(resolved) as connection:
        connection.execute(
            "INSERT INTO raw_snapshots("
            "snapshot_id,provider,kind,path,headers_path,retrieved_at_utc,content_hash,byte_count) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                snapshot_id,
                "the-odds-api",
                "full-board",
                str(raw_path),
                str(headers_path),
                retrieved_at,
                content_hash,
                len(response.content),
            ),
        )
        connection.execute(
            "UPDATE api_requests SET completed_at_utc=?,status=?,http_status=?,"
            "provider_requests_used=?,provider_requests_remaining=?,provider_requests_last=?,"
            "raw_snapshot_id=? WHERE request_id=?",
            (
                retrieved_at,
                "RAW_SAVED",
                response.status_code,
                _safe_int_header(response.headers, "x-requests-used"),
                _safe_int_header(response.headers, "x-requests-remaining"),
                _safe_int_header(response.headers, "x-requests-last"),
                snapshot_id,
                request_id,
            ),
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE api_requests SET status='HTTP_FAILED',error_message=? WHERE request_id=?",
                (_redact(str(exc), api_key)[:2000], request_id),
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
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE api_requests SET status='PARSE_FAILED',error_message=? WHERE request_id=?",
                (str(exc)[:2000], request_id),
            )
        raise
    with transaction(resolved) as connection:
        connection.execute(
            "UPDATE api_requests SET status='COMPLETE' WHERE request_id=?", (request_id,)
        )
    return {
        "request_id": request_id,
        "snapshot_id": snapshot_id,
        "week_bucket": week,
        "events": len(payload),
        "quotes": len(parsed.quotes),
        "consensus_markets": len(parsed.consensus),
        "provider_credit_cost": _safe_int_header(response.headers, "x-requests-last"),
        "raw_path": raw_path,
    }
