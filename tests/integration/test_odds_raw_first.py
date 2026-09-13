from __future__ import annotations

import json

import httpx
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import snapshot_odds
from nfl_bets.odds.consensus import ZeroJuiceFlatlineError


def test_raw_response_is_saved_before_flatline_parse_failure(tmp_path, monkeypatch) -> None:
    settings = Settings(root=tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key-never-logged")
    board = [
        {
            "id": "event-1",
            "commence_time": "2026-09-14T00:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "New York Jets",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "last_update": "2026-09-13T12:00:00Z",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 44.5, "price": 100},
                                {"name": "Under", "point": 44.5, "price": 100},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=board, headers={"x-requests-last": "2"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ZeroJuiceFlatlineError):
        snapshot_odds("manual", settings=settings, client=client)
    raw_files = list((settings.raw_dir / "odds").rglob("*.json"))
    body_files = [path for path in raw_files if not path.name.endswith(".headers.json")]
    assert len(body_files) == 1
    assert json.loads(body_files[0].read_text(encoding="utf-8")) == board
    with connect(settings) as connection:
        request = connection.execute("SELECT status,raw_snapshot_id FROM api_requests").fetchone()
    assert request["status"] == "PARSE_FAILED"
    assert request["raw_snapshot_id"] is not None
