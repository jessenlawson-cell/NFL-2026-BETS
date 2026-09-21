from __future__ import annotations

import pytest

from scripts.shadow_predict import (
    contract_candidates,
    evidence_summary,
    football_evidence,
    injury_overlay,
)


@pytest.mark.parametrize(
    ("market", "selection", "raw", "weight", "source", "direction"),
    [
        ("spreads", "HOME", 2.0, 0.0, "MARKET_CALIBRATION_ONLY", "NEUTRAL"),
        ("spreads", "HOME", 2.0, 0.5, "FOOTBALL_ADJUSTED", "SUPPORTS_SELECTION"),
        ("spreads", "AWAY", 2.0, 0.5, "FOOTBALL_ADJUSTED", "OPPOSES_SELECTION"),
        ("totals", "OVER", -2.0, 0.5, "FOOTBALL_ADJUSTED", "OPPOSES_SELECTION"),
        ("totals", "UNDER", -2.0, 0.5, "FOOTBALL_ADJUSTED", "SUPPORTS_SELECTION"),
    ],
)
def test_football_evidence_reports_weighted_direction(
    market: str,
    selection: str,
    raw: float,
    weight: float,
    source: str,
    direction: str,
) -> None:
    assert football_evidence(market, selection, raw, weight) == {
        "weighted_adjustment": raw * weight,
        "signal_source": source,
        "football_direction": direction,
    }


def test_injury_overlay_preserves_canonical_home_and_away_diagnostics() -> None:
    rows = [
        {
            "team_id": "BUF",
            "is_home": False,
            "qb_out_doubtful": 1,
            "ol_out_doubtful": 2,
            "defensive_front_out_doubtful": 3,
            "secondary_out_doubtful": 4,
            "questionable_players": 5,
        },
        {
            "team_id": "NYJ",
            "is_home": True,
            "qb_out_doubtful": 0,
            "ol_out_doubtful": 1,
            "defensive_front_out_doubtful": 0,
            "secondary_out_doubtful": 2,
            "questionable_players": 3,
        },
    ]

    assert injury_overlay(rows) == {
        "injury_data_status": "COMPLETE_UNWEIGHTED",
        "away_injuries": {
            "team_id": "BUF",
            "qb_out_doubtful": 1,
            "ol_out_doubtful": 2,
            "defensive_front_out_doubtful": 3,
            "secondary_out_doubtful": 4,
            "questionable_players": 5,
        },
        "home_injuries": {
            "team_id": "NYJ",
            "qb_out_doubtful": 0,
            "ol_out_doubtful": 1,
            "defensive_front_out_doubtful": 0,
            "secondary_out_doubtful": 2,
            "questionable_players": 3,
        },
    }


def test_evidence_summary_counts_qualifying_candidate_evidence() -> None:
    candidates = [
        {
            "signal_source": "MARKET_CALIBRATION_ONLY",
            "football_direction": "NEUTRAL",
        },
        {
            "signal_source": "FOOTBALL_ADJUSTED",
            "football_direction": "SUPPORTS_SELECTION",
        },
        {
            "signal_source": "FOOTBALL_ADJUSTED",
            "football_direction": "OPPOSES_SELECTION",
        },
    ]

    assert evidence_summary(candidates) == {
        "signal_source_counts": {
            "MARKET_CALIBRATION_ONLY": 1,
            "FOOTBALL_ADJUSTED": 2,
        },
        "football_direction_counts": {
            "SUPPORTS_SELECTION": 1,
            "OPPOSES_SELECTION": 1,
            "NEUTRAL": 1,
        },
    }


def test_contract_candidates_preserve_push_aware_ev_and_non_push_edge() -> None:
    scores = {
        "raw_adjustment": 2.0,
        "adjustment_weight": 0.5,
        "final_projection": 4.0,
        "calibrated_non_push_win_probability": 0.60,
        "model_win_probability": 0.54,
        "model_push_probability": 0.10,
        "model_loss_probability": 0.36,
    }
    quotes = [
        {
            "selection": "HOME",
            "point": -3.0,
            "american_price": -110,
            "vig_free_probability": 0.50,
        },
        {
            "selection": "AWAY",
            "point": 3.0,
            "american_price": -110,
            "vig_free_probability": 0.50,
        },
    ]

    home, away = contract_candidates(
        game_id="game",
        game="BUF at NYJ",
        market="spreads",
        line=3.0,
        book="pinnacle",
        scores=scores,
        quotes=quotes,
        injuries={
            "injury_data_status": "COMPLETE_UNWEIGHTED",
            "away_injuries": {"team_id": "BUF"},
            "home_injuries": {"team_id": "NYJ"},
        },
        feature_as_of="2026-09-20T15:42:19+00:00",
    )

    assert home["model_probability"] == pytest.approx(0.60)
    assert home["weighted_adjustment"] == pytest.approx(1.0)
    assert home["signal_source"] == "FOOTBALL_ADJUSTED"
    assert home["football_direction"] == "SUPPORTS_SELECTION"
    assert home["injury_data_status"] == "COMPLETE_UNWEIGHTED"
    assert home["feature_as_of"] == "2026-09-20T15:42:19+00:00"
    assert home["offered_point"] == pytest.approx(-3.0)
    assert home["probability_edge"] == pytest.approx(0.10)
    assert home["expected_roi"] == pytest.approx(0.54 * (100 / 110) - 0.36)
    assert home["push_probability"] == pytest.approx(0.10)
    assert away["model_probability"] == pytest.approx(0.40)
    assert away["football_direction"] == "OPPOSES_SELECTION"
    assert away["offered_point"] == pytest.approx(3.0)
    assert away["probability_edge"] == pytest.approx(-0.10)
    assert away["expected_roi"] == pytest.approx(0.36 * (100 / 110) - 0.54)
