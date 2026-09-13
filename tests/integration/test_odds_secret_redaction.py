from __future__ import annotations

import httpx
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import snapshot_odds


def test_provider_failure_never_persists_api_key(tmp_path, monkeypatch) -> None:
    secret = "fixture-secret-key"
    monkeypatch.setenv("THE_ODDS_API_KEY", secret)
    settings = Settings(root=tmp_path)
    initialize_database(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "unauthorized"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="raw response was saved"):
        snapshot_odds("manual", settings=settings, client=client)
    with connect(settings) as connection:
        stored = connection.execute("SELECT error_message FROM api_requests").fetchone()[0]
    assert secret not in stored
