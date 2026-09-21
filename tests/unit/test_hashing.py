from __future__ import annotations

from nfl_bets.util import canonical_hash


def test_retrieval_time_does_not_change_content_hash() -> None:
    first = {
        "game_id": "2025_01_A_B",
        "source": "fixture",
        "schema_version": "1.0.0",
        "retrieved_at_utc": "2026-01-01T00:00:00Z",
    }
    second = {**first, "retrieved_at_utc": "2026-01-02T00:00:00Z"}
    assert canonical_hash(first) == canonical_hash(second)


def test_business_data_change_changes_content_hash() -> None:
    first = {"game_id": "2025_01_A_B", "home_score": 20, "source": "fixture"}
    second = {**first, "home_score": 21}
    assert canonical_hash(first) != canonical_hash(second)
