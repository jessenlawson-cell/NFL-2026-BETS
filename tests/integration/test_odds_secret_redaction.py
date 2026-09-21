from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import snapshot_odds


def test_provider_failure_never_persists_api_key(tmp_path, monkeypatch) -> None:
    secret = "fixture-secret-key"
    monkeypatch.setenv("THE_ODDS_API_KEY", secret)
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"message": "unauthorized"},
            headers={
                "set-cookie": f"provider_session={secret}",
                "authorization": f"Bearer {secret}",
                "x-request-id": "safe-request-id",
                "x-requests-remaining": "499",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="raw response was saved"):
        snapshot_odds("manual", settings=settings, client=client)
    with connect(settings) as connection:
        row = connection.execute(
            "SELECT error_message,raw_snapshot_id FROM api_requests"
        ).fetchone()
        stored = row["error_message"]
        headers_path = connection.execute(
            "SELECT headers_path FROM raw_snapshots WHERE snapshot_id=?",
            (row["raw_snapshot_id"],),
        ).fetchone()[0]
    assert secret not in stored
    headers = json.loads(Path(headers_path).read_text(encoding="utf-8"))
    assert headers["x-request-id"] == "safe-request-id"
    assert headers["x-requests-remaining"] == "499"
    assert "set-cookie" not in headers
    assert "authorization" not in headers
    assert secret not in json.dumps(headers)
