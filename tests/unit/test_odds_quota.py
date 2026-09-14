from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import initialize_database, transaction
from nfl_bets.odds.client import QuotaError, _persist_parsed, snapshot_odds
from nfl_bets.odds.consensus import ParsedBoard


def test_provider_credit_preflight_blocks_two_market_request(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key")
    with transaction(settings) as connection:
        connection.execute(
            "INSERT INTO api_requests(request_id,slot,request_kind,week_bucket,started_at_utc,"
            "completed_at_utc,status,provider_requests_remaining) VALUES (?,?,?,?,?,?,?,?)",
            (
                "prior",
                "manual",
                "full-board",
                "2026-09-08",
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
                "COMPLETE",
                1,
            ),
        )
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(QuotaError, match="fewer than two credits"):
        snapshot_odds("manual", settings=settings, client=client)
    assert called is False


def test_market_export_is_promoted_once(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    writes: list[object] = []
    monkeypatch.setattr(
        "nfl_bets.odds.client.atomic_write_text",
        lambda path, text: writes.append(path),
    )
    _persist_parsed(ParsedBoard(quotes=[], consensus=[]), settings)
    assert writes == [settings.root / "market_odds.csv"]
