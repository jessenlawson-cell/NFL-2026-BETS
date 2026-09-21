from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_bets.research_sidecar import (
    build_qb_net_leverage,
    evaluate_matched_ablation,
    game_qb_net_leverage_diff,
    standardize_ablation_fold,
)

DECISION_TIME = datetime(2026, 10, 5, 16, tzinfo=UTC)
AVAILABLE_TIME = datetime(2026, 10, 5, 12, tzinfo=UTC)


def _frames() -> dict[str, pl.DataFrame]:
    schedules = pl.DataFrame(
        [
            {
                "game_id": f"2026_0{week}_AAA_BBB",
                "season": 2026,
                "week": week,
                "game_type": "REG",
                "gameday": f"2026-09-{week * 7:02d}",
                "gametime": "13:00",
                "away_team": "AAA",
                "home_team": "BBB",
                "away_score": 20,
                "home_score": 21,
            }
            for week in range(1, 5)
        ]
        + [
            {
                "game_id": "2026_06_AAA_BBB",
                "season": 2026,
                "week": 6,
                "game_type": "REG",
                "gameday": "2026-10-12",
                "gametime": "13:00",
                "away_team": "AAA",
                "home_team": "BBB",
                "away_score": 30,
                "home_score": 10,
            }
        ]
    )
    injuries = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [5, 6],
            "team": ["AAA", "AAA"],
            "gsis_id": ["starter-gsis", "backup-gsis"],
            "position": ["QB", "QB"],
            "report_status": ["Out", "Out"],
        }
    )
    depth_charts = pl.DataFrame(
        {
            "dt": [
                "2026-10-05T11:00:00Z",
                "2026-10-05T11:00:00Z",
                "2026-10-05T18:00:00Z",
                "2026-10-05T18:00:00Z",
            ],
            "team": ["AAA", "AAA", "AAA", "AAA"],
            "gsis_id": ["starter-gsis", "backup-gsis", "backup-gsis", "starter-gsis"],
            "pos_abb": ["QB", "QB", "QB", "QB"],
            "pos_rank": [1, 2, 1, 2],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["starter-gsis", "backup-gsis"],
            "pfr_id": ["Starter00", "Backup00"],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": [f"2026_0{week}_AAA_BBB" for week in range(1, 5)]
            + ["2026_06_AAA_BBB"],
            "pfr_player_id": ["Starter00"] * 5,
            "team": ["AAA"] * 5,
            "offense_pct": [1.0, 0.9, 0.8, 0.7, 0.0],
        }
    )
    pbp = pl.DataFrame(
        {
            "game_id": [
                "2026_01_AAA_BBB",
                "2026_02_AAA_BBB",
                "2026_03_AAA_BBB",
                "2026_04_AAA_BBB",
                "2026_04_AAA_BBB",
                "2026_06_AAA_BBB",
            ],
            "passer_player_id": ["starter-gsis"] * 4 + ["backup-gsis", "backup-gsis"],
            "qb_dropback": [1, 1, 1, 1, 1, 1],
            "epa": [0.4, 0.4, 0.4, 0.4, -0.2, 100.0],
        }
    )
    return {
        "schedules": schedules,
        "injuries": injuries,
        "depth_charts": depth_charts,
        "players": players,
        "snap_counts": snap_counts,
        "pbp": pbp,
    }


def _build(frames: dict[str, pl.DataFrame]) -> dict[str, object]:
    return build_qb_net_leverage(
        **frames,
        season=2026,
        week=5,
        team_id="AAA",
        decision_time=DECISION_TIME,
        source_available_at=AVAILABLE_TIME,
        half_life=4.0,
        prior_strength=0.0,
    )


def test_qb_shock_uses_latest_predecision_depth_and_prior_games() -> None:
    result = _build(_frames())

    assert result["eligibility_status"] == "ELIGIBLE_RESEARCH_ONLY"
    assert result["starter_player_gsis_id"] == "starter-gsis"
    assert result["backup_player_gsis_id"] == "backup-gsis"
    assert result["lost_snap_share_4g"] == pytest.approx(0.85)
    assert result["starter_pregame_epa_per_dropback"] == pytest.approx(0.4)
    assert result["backup_pregame_epa_per_dropback"] == pytest.approx(-0.2)
    assert result["qb_net_le_shock"] == pytest.approx(-0.51)


def test_no_qualifying_out_event_is_zero_not_missing() -> None:
    frames = _frames()
    frames["injuries"] = frames["injuries"].with_columns(
        pl.lit("Questionable").alias("report_status")
    )

    result = _build(frames)

    assert result["eligibility_status"] == "NO_QUALIFYING_EVENT"
    assert result["qb_net_le_shock"] == 0.0


def test_missing_team_week_injury_coverage_is_null_not_zero() -> None:
    frames = _frames()
    frames["injuries"] = frames["injuries"].head(0)

    result = _build(frames)

    assert result["eligibility_status"] == "INELIGIBLE_MISSING_INJURY_COVERAGE"
    assert result["qb_net_le_shock"] is None


def test_qualifying_event_with_no_backup_rate_remains_null() -> None:
    frames = _frames()
    frames["pbp"] = frames["pbp"].filter(pl.col("passer_player_id") != "backup-gsis")

    result = _build(frames)

    assert result["eligibility_status"] == "INELIGIBLE_MISSING_BACKUP_RATE"
    assert result["qb_net_le_shock"] is None


def test_postdecision_source_is_rejected() -> None:
    frames = _frames()
    with pytest.raises(ValueError, match="source_available_at must not exceed decision_time"):
        build_qb_net_leverage(
            **frames,
            season=2026,
            week=5,
            team_id="AAA",
            decision_time=DECISION_TIME,
            source_available_at=datetime(2026, 10, 5, 17, tzinfo=UTC),
            half_life=4.0,
            prior_strength=0.0,
        )


def test_duplicate_crosswalk_ids_fail_closed() -> None:
    frames = _frames()
    frames["players"] = pl.concat([frames["players"], frames["players"].head(1)])

    with pytest.raises(ValueError, match="Duplicate GSIS or PFR identifiers"):
        _build(frames)


def test_fold_standardization_fits_train_only_and_preserves_null() -> None:
    train = pl.DataFrame({"qb_net_le_shock": [0.0, 2.0]})
    validation = pl.DataFrame({"qb_net_le_shock": [100.0, None]})

    train_scaled, validation_scaled = standardize_ablation_fold(train, validation)

    assert train_scaled["qb_net_le_shock_z"].to_list() == [-1.0, 1.0]
    assert validation_scaled["qb_net_le_shock_z"][0] == pytest.approx(99.0)
    assert validation_scaled["qb_net_le_shock_z"][1] is None


def test_game_difference_is_null_when_a_qualifying_side_is_ineligible() -> None:
    assert game_qb_net_leverage_diff(
        {"qb_net_le_shock": -0.2}, {"qb_net_le_shock": None}
    ) is None


def test_ablation_evaluation_is_matched_pass_only_evidence() -> None:
    observations = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "actual": [1, 0, 1, 0],
            "baseline_probability": [0.6, 0.4, 0.6, 0.4],
            "candidate_probability": [0.8, 0.2, 0.8, 0.2],
            "qb_net_le_shock": [-0.2, 0.0, -0.3, 0.0],
            "evidence_status": ["PROSPECTIVE_ELIGIBLE"] * 4,
        }
    )

    result = evaluate_matched_ablation(observations, bootstrap_samples=200, seed=7)

    assert result["status"] == "RESEARCH_ONLY_UNWEIGHTED"
    assert result["decision"] == "PASS"
    assert result["rows"] == 4
    assert result["delta"]["brier"] == pytest.approx(-0.12)
    assert result["paired_bootstrap_95pct"]["brier_delta"][1] < 0
    assert result["paired_bootstrap_95pct"]["log_loss_delta"][1] < 0
    assert result["meets_research_acceptance"] is True
    assert {"stake", "kelly", "expected_roi"}.isdisjoint(result)


def test_ablation_evaluation_rejects_unmatched_null_rows() -> None:
    observations = pl.DataFrame(
        {
            "game_id": ["g1"],
            "actual": [1],
            "baseline_probability": [0.6],
            "candidate_probability": [None],
            "qb_net_le_shock": [-0.2],
            "evidence_status": ["PROSPECTIVE_ELIGIBLE"],
        }
    )

    with pytest.raises(ValueError, match="matched non-null rows"):
        evaluate_matched_ablation(observations, bootstrap_samples=20, seed=7)


def test_ablation_evaluation_rejects_backfilled_evidence() -> None:
    observations = pl.DataFrame(
        {
            "game_id": ["g1"],
            "actual": [1],
            "baseline_probability": [0.6],
            "candidate_probability": [0.8],
            "qb_net_le_shock": [-0.2],
            "evidence_status": ["RESEARCH_BACKFILL_NOT_OOS_ELIGIBLE"],
        }
    )

    with pytest.raises(ValueError, match="prospectively eligible evidence"):
        evaluate_matched_ablation(observations, bootstrap_samples=20, seed=7)
