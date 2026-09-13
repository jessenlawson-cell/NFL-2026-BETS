from __future__ import annotations

import pytest

from nfl_bets.odds.consensus import (
    ZeroJuiceFlatlineError,
    american_to_decimal,
    build_consensus,
    parse_board,
    remove_vig,
)


def _quote(book: str, line: float, probability: float = 0.5) -> dict[str, object]:
    return {
        "snapshot_id": "snapshot",
        "provider_event_id": "event",
        "game_id": "game",
        "market": "spreads",
        "selection": "HOME",
        "bookmaker_key": book,
        "canonical_line": line,
        "vig_free_probability": probability,
        "retrieved_at_utc": "2026-09-13T12:00:00Z",
        "source_updated_at_utc": None,
    }


def test_american_conversion_and_vig_removal() -> None:
    assert american_to_decimal(-110) == pytest.approx(1.909090909)
    assert american_to_decimal(150) == pytest.approx(2.5)
    fair, overround = remove_vig([-110, -110])
    assert overround > 1.0
    assert fair == pytest.approx([0.5, 0.5])


def test_zero_juice_flatline_is_rejected() -> None:
    with pytest.raises(ZeroJuiceFlatlineError, match="overround"):
        remove_vig([100, 100])


def test_spread_signs_normalize_to_projected_home_margin() -> None:
    payload = [
        {
            "id": "event",
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
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Buffalo Bills", "point": -3.5, "price": -110},
                                {"name": "New York Jets", "point": 3.5, "price": -110},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    parsed = parse_board(
        payload,
        "snapshot",
        "2026-09-13T12:05:00Z",
        "1.0.0",
        {"event": "2026_01_NYJ_BUF"},
    )
    assert {quote["canonical_line"] for quote in parsed.quotes} == {3.5}


def test_same_point_consensus_blends_line_and_probability() -> None:
    consensus = build_consensus(
        [
            _quote("pinnacle", 3.0, 0.52),
            _quote("draftkings", 3.0, 0.50),
            _quote("fanduel", 3.0, 0.48),
        ],
        "1.0.0",
    )[0]
    assert consensus["status"] == "VALID"
    assert consensus["consensus_line"] == pytest.approx(3.0)
    assert consensus["consensus_probability"] == pytest.approx(0.508)


def test_different_points_never_average_probability_and_log_dislocation() -> None:
    consensus = build_consensus(
        [_quote("pinnacle", 3.0, 0.52), _quote("draftkings", 3.5), _quote("fanduel", 4.0)],
        "1.0.0",
    )[0]
    assert consensus["status"] == "INCOMPARABLE_POINTS"
    assert consensus["consensus_probability"] is None
    assert consensus["consensus_line"] == pytest.approx(3.3)
    assert "draftkings" in consensus["dislocated_books_json"]


def test_missing_pinnacle_is_retail_only() -> None:
    consensus = build_consensus([_quote("draftkings", 2.5), _quote("fanduel", 2.5)], "1.0.0")[0]
    assert consensus["status"] == "RETAIL_ONLY"
    assert consensus["consensus_line"] is None
    assert consensus["consensus_probability"] is None


def test_stale_pinnacle_falls_back_to_retail_diagnostics() -> None:
    rows = [
        _quote("pinnacle", 3.0, 0.52),
        _quote("draftkings", 3.0, 0.50),
        _quote("fanduel", 3.0, 0.48),
    ]
    for row in rows:
        row["source_updated_at_utc"] = "2026-09-13T11:50:00Z"
    rows[0]["source_updated_at_utc"] = "2026-09-13T10:00:00Z"
    consensus = build_consensus(rows, "1.0.0", freshness_minutes=30)[0]
    assert consensus["status"] == "RETAIL_ONLY"
    assert consensus["anchor_line"] is None
    assert consensus["retail_books_count"] == 2
