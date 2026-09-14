from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from nfl_bets.model.residuals import OutcomeProbabilities
from nfl_bets.prospective import (
    ELIGIBLE_CLOSE_STATUS,
    _anchor_contract,
    _score_contract,
    select_canonical_prediction,
)


def _policy() -> dict[str, object]:
    return {
        "close_window_minutes_before_kickoff": {"earliest": 90, "latest": 5},
        "quote_max_age_minutes": 30,
    }


def _prediction(snapshot_time: str, prediction_id: str = "prediction") -> dict[str, object]:
    return {
        "prediction_id": prediction_id,
        "eligibility_status": ELIGIBLE_CLOSE_STATUS,
        "snapshot_retrieved_at_utc": snapshot_time,
        "prediction_created_at_utc": snapshot_time,
        "pinnacle_updated_at_utc": snapshot_time,
        "pinnacle_line": 3.0,
        "pinnacle_orientation_no_vig_probability": 0.5,
        "calibrated_non_push_win_probability": 0.5,
    }


def test_canonical_selection_uses_latest_valid_snapshot_without_outcomes() -> None:
    kickoff = datetime(2026, 9, 20, 17, 0, tzinfo=UTC)
    early = _prediction("2026-09-20T15:45:00Z", "early")
    latest = _prediction("2026-09-20T16:45:00Z", "latest")
    selected = select_canonical_prediction([latest, early], kickoff, _policy())
    assert selected is not None
    assert selected["prediction_id"] == "latest"

    # Outcome-like values are irrelevant to canonical selection.
    early["actual_value"] = 100
    latest["actual_value"] = -100
    selected_again = select_canonical_prediction([latest, early], kickoff, _policy())
    assert selected_again is not None
    assert selected_again["prediction_id"] == "latest"


def test_canonical_selection_enforces_window_and_quote_age() -> None:
    kickoff = datetime(2026, 9, 20, 17, 0, tzinfo=UTC)
    too_early = _prediction("2026-09-20T15:29:59Z")
    stale = _prediction("2026-09-20T16:00:00Z")
    stale["pinnacle_updated_at_utc"] = "2026-09-20T15:29:59Z"
    assert select_canonical_prediction([too_early, stale], kickoff, _policy()) is None


def test_anchor_contract_rejects_stale_or_different_points() -> None:
    snapshot = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
    rows = [
        {
            "bookmaker_key": "pinnacle",
            "selection": "HOME",
            "canonical_line": 3.0,
            "american_price": -110,
            "vig_free_probability": 0.5,
            "overround": 1.0476,
            "source_updated_at_utc": "2026-09-20T15:20:00Z",
        },
        {
            "bookmaker_key": "pinnacle",
            "selection": "AWAY",
            "canonical_line": 3.0,
            "american_price": -110,
            "vig_free_probability": 0.5,
            "overround": 1.0476,
            "source_updated_at_utc": "2026-09-20T15:20:00Z",
        },
    ]
    assert _anchor_contract(rows, "spreads", snapshot, 30).reason == "PINNACLE_QUOTE_STALE"
    rows[1]["canonical_line"] = 3.5
    assert _anchor_contract(rows, "spreads", snapshot, 60).reason == "PINNACLE_POINTS_DIFFER"


def test_probability_scoring_recombines_calibrated_non_push_and_push_mass() -> None:
    class Model:
        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.asarray([2.0])

    class Mapper:
        def probabilities(
            self, projection: float, line: float, stratum: float
        ) -> OutcomeProbabilities:
            return OutcomeProbabilities(win=0.45, push=0.10, loss=0.45)

    class Calibrator:
        def predict(self, probabilities: np.ndarray) -> np.ndarray:
            return np.asarray([0.60])

    candidate = SimpleNamespace(
        spread_adjustment_model=Model(),
        spread_adjustment_weight=0.5,
        spread_residuals=Mapper(),
        spread_calibrator=Calibrator(),
    )
    scored = _score_contract(candidate, "spreads", 3.0, np.zeros((1, 21)))
    assert scored["final_projection"] == pytest.approx(4.0)
    assert scored["model_win_probability"] == pytest.approx(0.54)
    assert scored["model_push_probability"] == pytest.approx(0.10)
    assert scored["model_loss_probability"] == pytest.approx(0.36)
    assert sum(
        scored[key]
        for key in (
            "model_win_probability",
            "model_push_probability",
            "model_loss_probability",
        )
    ) == pytest.approx(1.0)
