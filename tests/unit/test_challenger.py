from __future__ import annotations

from datetime import UTC, datetime

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typer.testing import CliRunner

from nfl_bets.challenger import (
    PreparedChallengerFeatures,
    _power_rankings,
    compare_challenger,
    latest_valid_decision_snapshot,
    rank_shadow_candidates,
    settle_challenger_predictions,
)
from nfl_bets.cli import app
from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.features.challenger import (
    CHALLENGER_METRICS,
    _advanced_observations,
    _validate_through_week,
)
from nfl_bets.model.challenger import (
    CHALLENGER_FEATURE_FAMILIES,
    OPPONENT_ADJUSTED_METRICS,
    ChallengerCandidate,
    ChallengerMarketModel,
    WeightedBlendRegressor,
    _candidate_feature_sets,
    _chronological_oof,
    _opponent_adjusted_observations,
    _paired_probability_bootstrap,
    _probability_oof,
    _validate_training_cutoff,
    assemble_challenger_matrix,
    build_matchup_frame,
    quantitative_drivers,
    score_challenger_contract,
)
from nfl_bets.model.residuals import ProbabilityCalibrator, StratifiedResidualMapper
from nfl_bets.prospective import ProspectiveDataError


def test_advanced_observations_use_wp_filter_and_keep_special_teams_separate() -> None:
    rows = [
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": 0.6,
            "success": 1.0,
            "wp": 0.50,
            "pass_oe": 0.20,
            "qb_dropback": 1.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "cpoe": 8.0,
            "air_epa": 0.4,
            "yac_epa": 0.2,
            "yards_gained": 20.0,
            "down": 1.0,
            "ydstogo": 10.0,
            "no_huddle": 1.0,
            "shotgun": 1.0,
            "special_teams_play": 0.0,
            "play_type": "pass",
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": -4.0,
            "success": 0.0,
            "wp": 0.99,
            "pass_oe": -0.50,
            "qb_dropback": 1.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 1.0,
            "qb_hit": 1.0,
            "cpoe": -30.0,
            "air_epa": -2.0,
            "yac_epa": -2.0,
            "yards_gained": -8.0,
            "down": 3.0,
            "ydstogo": 12.0,
            "no_huddle": 0.0,
            "shotgun": 1.0,
            "special_teams_play": 0.0,
            "play_type": "pass",
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": 0.3,
            "success": 1.0,
            "wp": 0.40,
            "pass_oe": -0.10,
            "qb_dropback": 0.0,
            "rush_attempt": 1.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "cpoe": None,
            "air_epa": None,
            "yac_epa": None,
            "yards_gained": 12.0,
            "down": 3.0,
            "ydstogo": 2.0,
            "no_huddle": 0.0,
            "shotgun": 0.0,
            "special_teams_play": 0.0,
            "play_type": "run",
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "A",
            "defteam": "B",
            "epa": 0.5,
            "success": 1.0,
            "wp": 0.60,
            "pass_oe": None,
            "qb_dropback": 0.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "cpoe": None,
            "air_epa": None,
            "yac_epa": None,
            "yards_gained": 0.0,
            "down": None,
            "ydstogo": None,
            "no_huddle": 0.0,
            "shotgun": 0.0,
            "special_teams_play": 1.0,
            "play_type": "punt",
        },
        {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "posteam": "B",
            "defteam": "A",
            "epa": -0.1,
            "success": 0.0,
            "wp": 0.50,
            "pass_oe": 0.05,
            "qb_dropback": 1.0,
            "rush_attempt": 0.0,
            "qb_kneel": 0.0,
            "sack": 0.0,
            "qb_hit": 0.0,
            "cpoe": -2.0,
            "air_epa": -0.1,
            "yac_epa": 0.0,
            "yards_gained": 4.0,
            "down": 2.0,
            "ydstogo": 8.0,
            "no_huddle": 0.0,
            "shotgun": 1.0,
            "special_teams_play": 0.0,
            "play_type": "pass",
        },
    ]

    result = _advanced_observations(pl.DataFrame(rows)).sort("team_id")
    team_a = result.row(0, named=True)

    assert result.height == 2
    assert team_a["off_wp_pass_epa"] == pytest.approx(0.6)
    assert team_a["off_neutral_proe"] == pytest.approx(0.05)
    assert team_a["off_pass_explosive_rate"] == pytest.approx(1.0)
    assert team_a["off_wp_rush_epa"] == pytest.approx(0.3)
    assert team_a["off_short_yard_success_rate"] == pytest.approx(1.0)
    assert team_a["off_qb_hit_rate"] == pytest.approx(0.5)
    assert team_a["off_early_down_pass_rate"] == pytest.approx(1.0)
    assert team_a["st_epa"] == pytest.approx(0.5)


def test_advanced_observations_reject_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        _advanced_observations(pl.DataFrame({"game_id": ["g"]}))


def test_through_week_rejects_later_completed_current_season_game() -> None:
    games = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_type": "REG",
                "kickoff_utc": "2026-09-20T17:00:00Z",
                "home_score": 20.0,
                "away_score": 17.0,
            },
            {
                "season": 2026,
                "week": 3,
                "game_type": "REG",
                "kickoff_utc": "2026-09-27T17:00:00Z",
                "home_score": 24.0,
                "away_score": 21.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="beyond requested through-week 2"):
        _validate_through_week(
            games,
            datetime(2026, 9, 28, tzinfo=UTC),
            through_week=2,
        )


def test_matchup_frame_builds_directional_and_total_features() -> None:
    rows = []
    for is_home, value in ((True, 2.0), (False, 1.0)):
        row: dict[str, object] = {
            "game_id": "2025_01_A_B",
            "season": 2025,
            "week": 1,
            "is_home": is_home,
        }
        row.update({metric: value for metric in (*CHALLENGER_METRICS, *OPPONENT_ADJUSTED_METRICS)})
        rows.append(row)
    games = pl.DataFrame(
        {
            "game_id": ["2025_01_A_B"],
            "home_rest": [7.0],
            "away_rest": [6.0],
            "roof": ["outdoors"],
            "surface": ["grass"],
        }
    )

    result = build_matchup_frame(pl.DataFrame(rows), games)

    assert result.height == 1
    assert result["wp_pass_epa_matchup_difference"][0] == pytest.approx(0.0)
    assert result["wp_pass_epa_matchup_sum"][0] == pytest.approx(6.0)
    assert result["st_epa_difference"][0] == pytest.approx(1.0)
    assert result["rest_difference"][0] == pytest.approx(1.0)
    assert result["is_outdoors"][0] == pytest.approx(1.0)
    assert result["is_grass"][0] == pytest.approx(1.0)


def test_feature_candidates_cover_every_registered_family_without_duplicates() -> None:
    candidates = _candidate_feature_sets()
    assert candidates
    assert len({columns for columns in candidates.values()}) == len(candidates)
    used = set().union(*(set(columns) for columns in candidates.values()))
    registered = set().union(*CHALLENGER_FEATURE_FAMILIES.values())
    assert used == registered


def _market_component(
    model: Pipeline | WeightedBlendRegressor,
    mapper: StratifiedResidualMapper,
    calibrator: ProbabilityCalibrator,
    *,
    football_weight: float = 1.0,
    family: str = "ridge",
) -> ChallengerMarketModel:
    return ChallengerMarketModel(
        feature_columns=("x",),
        model_family=family,
        model=model,
        ridge_model=model,
        tree_model=model,
        football_weight=football_weight,
        residuals=mapper,
        calibrator=calibrator,
        probability_margin=0.03,
        selected_half_life=2.0,
        selected_prior_strength=1.0,
        selected_feature_set="test",
        feature_mean=np.asarray([0.0]),
        feature_std=np.asarray([1.0]),
        feature_min=np.asarray([-1.0]),
        feature_max=np.asarray([1.0]),
    )


def test_weighted_blend_is_deterministic_and_uses_registered_weight() -> None:
    ridge = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.0))])
    tree = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.0))])
    x = np.asarray([[0.0], [1.0], [2.0]])
    ridge.fit(x, np.asarray([0.0, 1.0, 2.0]))
    tree.fit(x, np.asarray([2.0, 3.0, 4.0]))
    blend = WeightedBlendRegressor(ridge, tree, ridge_weight=0.25)

    first = blend.predict(np.asarray([[1.0]]))
    second = blend.predict(np.asarray([[1.0]]))

    assert first == pytest.approx(np.asarray([2.5]))
    assert second == pytest.approx(first)


def test_challenger_contract_market_regresses_and_reports_uncertainty() -> None:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.0))])
    model.fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([2.0, 3.0, 4.0]))
    actual = np.asarray([float(value) for value in range(-15, 15)] * 2)
    projection = np.zeros_like(actual)
    mapper = StratifiedResidualMapper(key_numbers=(3, 6, 7, 10, 14), absolute_strata=True)
    mapper.fit(actual, projection, np.zeros_like(actual))
    raw = np.asarray([mapper.probabilities(0.0, 0.0, 0.0).win] * 30)
    outcomes = np.asarray([0, 1] * 15)
    calibrator = ProbabilityCalibrator().fit(raw, outcomes)
    component = _market_component(
        model,
        mapper,
        calibrator,
        football_weight=0.25,
    )
    candidate = ChallengerCandidate(
        version="challenger-test",
        spread=component,
        total=component,
        training_cutoff="2026-W02",
        prospective_start_week=3,
        feature_hash="feature",
        spec_hash="spec",
    )

    result = score_challenger_contract(candidate, "spreads", 3.0, np.asarray([[2.0]]))

    assert result["raw_football_projection"] == pytest.approx(4.0)
    assert result["final_projection"] == pytest.approx(3.25)
    assert result["raw_adjustment"] == pytest.approx(1.0)
    assert result["adjustment_weight"] == pytest.approx(0.25)
    assert result["selected_model_family"] == "ridge"
    assert result["component_disagreement"] == pytest.approx(0.0)
    assert result["feature_drift_max_abs_z"] == pytest.approx(2.0)
    assert result["feature_outside_training_range"] == 1
    assert result["probability_margin"] == pytest.approx(0.03)
    assert result["model_win_probability"] + result["model_push_probability"] + result[
        "model_loss_probability"
    ] == pytest.approx(1.0)
    assert (
        quantitative_drivers(candidate, "spreads", np.asarray([[2.0]]), top=1)[0]["feature"] == "x"
    )


def test_zero_football_weight_returns_the_vig_free_market_probability() -> None:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.0))])
    model.fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([2.0, 3.0, 4.0]))
    mapper = StratifiedResidualMapper(key_numbers=(3, 6, 7, 10, 14))
    mapper.fit(np.asarray([-1.0, 1.0] * 15), np.zeros(30), np.zeros(30))
    calibrator = ProbabilityCalibrator().fit(np.asarray([0.4, 0.6] * 15), np.asarray([0, 1] * 15))
    component = _market_component(model, mapper, calibrator, football_weight=0.0)
    candidate = ChallengerCandidate(
        version="challenger-test",
        spread=component,
        total=component,
        training_cutoff="2026-W02",
        prospective_start_week=3,
        feature_hash="feature",
        spec_hash="spec",
    )

    result = score_challenger_contract(
        candidate,
        "spreads",
        3.0,
        np.asarray([[2.0]]),
        market_probability=0.5275,
    )

    assert result["calibrated_non_push_win_probability"] == pytest.approx(0.5275)
    assert result["signal_source"] == "MARKET_CALIBRATION_ONLY"


def test_challenger_candidate_serialization_round_trip(tmp_path) -> None:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(np.asarray([[-1.0], [0.0], [1.0]]), np.asarray([-1.0, 0.0, 1.0]))
    mapper = StratifiedResidualMapper().fit(
        np.asarray([-1.0, 1.0] * 15), np.zeros(30), np.zeros(30)
    )
    calibrator = ProbabilityCalibrator().fit(np.asarray([0.4, 0.6] * 15), np.asarray([0, 1] * 15))
    component = _market_component(model, mapper, calibrator)
    candidate = ChallengerCandidate(
        version="challenger-test",
        spread=component,
        total=component,
        training_cutoff="2026-W02",
        prospective_start_week=3,
        feature_hash="feature",
        spec_hash="spec",
    )
    path = tmp_path / "candidate.joblib"

    joblib.dump(candidate, path)
    restored = joblib.load(path)

    assert score_challenger_contract(
        restored, "spreads", 0.0, np.asarray([[0.5]])
    ) == score_challenger_contract(candidate, "spreads", 0.0, np.asarray([[0.5]]))


def test_paired_probability_bootstrap_is_reproducible() -> None:
    outcome = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    model = np.asarray([0.40, 0.70, 0.65, 0.30, 0.60, 0.45])
    market = np.asarray([0.48, 0.52, 0.51, 0.49, 0.50, 0.50])

    first = _paired_probability_bootstrap(outcome, model, market)
    second = _paired_probability_bootstrap(outcome, model, market)

    assert first == second
    assert first["samples"] == 1000


def test_power_rankings_use_the_frozen_projection_on_a_neutral_matchup() -> None:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.0))])
    model.fit(np.asarray([[-1.0], [0.0], [1.0]]), np.asarray([-7.0, 0.0, 7.0]))
    mapper = StratifiedResidualMapper(key_numbers=(3, 6, 7, 10, 14))
    mapper.fit(np.asarray([-1.0, 1.0] * 15), np.zeros(30), np.zeros(30))
    calibrator = ProbabilityCalibrator().fit(np.asarray([0.4, 0.6] * 15), np.asarray([0, 1] * 15))
    component = _market_component(model, mapper, calibrator)
    component.feature_columns = ("st_epa_difference",)
    candidate = ChallengerCandidate(
        version="challenger-test",
        spread=component,
        total=component,
        training_cutoff="2026-W02",
        prospective_start_week=3,
        feature_hash="feature",
        spec_hash="spec",
    )
    rows = []
    for team_id, st_epa in (("A", 1.0), ("B", -1.0)):
        row: dict[str, object] = {
            "game_id": f"next_{team_id}",
            "team_id": team_id,
            "kickoff_dt": datetime(2026, 9, 27, tzinfo=UTC),
        }
        row.update({metric: 0.0 for metric in (*CHALLENGER_METRICS, *OPPONENT_ADJUSTED_METRICS)})
        row["st_epa"] = st_epa
        rows.append(row)
    prepared = PreparedChallengerFeatures(
        lags=pl.DataFrame(),
        materialized=pl.DataFrame(rows),
        games=pl.DataFrame(),
        as_of=datetime(2026, 9, 22, tzinfo=UTC),
        input_hash="feature",
    )

    rankings = _power_rankings(prepared, candidate)

    assert [row["team_id"] for row in rankings] == ["A", "B"]
    assert rankings[0]["neutral_field_rating"] > rankings[1]["neutral_field_rating"]


def test_chronological_oof_never_scores_the_first_training_season() -> None:
    rows = [
        {
            "game_id": f"{season}_{index}",
            "season": season,
            "x": float(index),
            "target_margin": float(index + season - 2020),
            "spread_line": 0.0,
        }
        for season in range(2021, 2025)
        for index in range(40)
    ]

    result = _chronological_oof(
        pl.DataFrame(rows),
        feature_columns=("x",),
        target="target_margin",
        line_column="spread_line",
        alpha=1.0,
    )

    assert result["season"].min() == 2022
    assert result["season"].max() == 2024
    assert result.height == 120
    assert result.select("game_id").is_duplicated().any() is False


def test_challenger_training_matrix_stops_at_registered_week(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    games = []
    team_rows = []
    for week in (2, 3):
        game_id = f"2026_{week:02d}_A_B"
        games.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "game_type": "REG",
                "kickoff_utc": f"2026-09-{13 + week:02d}T17:00:00Z",
                "away_team": "A",
                "home_team": "B",
                "away_score": 17.0,
                "home_score": 20.0,
                "home_rest": 7.0,
                "away_rest": 7.0,
                "spread_line": 2.5,
                "total_line": 42.5,
                "roof": "outdoors",
                "surface": "grass",
            }
        )
        for is_home, value in ((True, 2.0), (False, 1.0)):
            row: dict[str, object] = {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "is_home": is_home,
            }
            row.update(
                {metric: value for metric in (*CHALLENGER_METRICS, *OPPONENT_ADJUSTED_METRICS)}
            )
            team_rows.append(row)
    pl.DataFrame(games).write_csv(settings.root / "games.csv")

    result = assemble_challenger_matrix(pl.DataFrame(team_rows), settings=settings, through_week=2)

    assert result["game_id"].to_list() == ["2026_02_A_B"]


def test_probability_oof_is_push_aware_and_bounded() -> None:
    rows = [
        {
            "game_id": f"{season}_{index}",
            "season": season,
            "x": float(index % 9) - 4.0,
            "target_margin": float((index % 9) - 4),
            "spread_line": 0.0,
        }
        for season in range(2021, 2025)
        for index in range(45)
    ]

    result = _probability_oof(
        pl.DataFrame(rows),
        feature_columns=("x",),
        target="target_margin",
        line_column="spread_line",
        alpha=1.0,
        key_numbers=(3, 6, 7, 10, 14),
    )

    assert result["season"].min() == 2022
    assert result["raw_non_push_probability"].is_between(0.0, 1.0).all()
    assert result["push_probability"].is_between(0.0, 1.0).all()
    assert result["football_weight"].is_between(0.0, 1.0).all()
    assert "raw_projection" in result.columns
    assert result.filter(pl.col("actual") == pl.col("line"))["is_push"].all()


def test_training_cutoff_requires_every_game_in_cutoff_week_to_be_final() -> None:
    games = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [2, 2],
            "game_type": ["REG", "REG"],
            "home_score": [20.0, None],
            "away_score": [17.0, None],
        }
    )
    with pytest.raises(ValueError, match="1 of 2 Week 2 games are final"):
        _validate_training_cutoff(games, through_week=2)


def test_opponent_adjustment_uses_only_pregame_opponent_strength() -> None:
    raw = pl.DataFrame(
        {
            "game_id": ["g", "g"],
            "season": [2025, 2025],
            "week": [2, 2],
            "team_id": ["A", "B"],
            "opponent_team_id": ["B", "A"],
            "off_wp_pass_epa": [0.30, 0.10],
            "def_wp_pass_epa": [0.05, 0.10],
            "off_wp_rush_epa": [0.20, 0.00],
            "def_wp_rush_epa": [0.00, 0.05],
            "off_wp_pass_success_rate": [0.55, 0.45],
            "def_wp_pass_success_rate": [0.42, 0.48],
            "off_wp_rush_success_rate": [0.52, 0.44],
            "def_wp_rush_success_rate": [0.43, 0.47],
        }
    )
    pregame = raw.select(
        "game_id",
        "team_id",
        "off_wp_pass_epa",
        "def_wp_pass_epa",
        "off_wp_rush_epa",
        "def_wp_rush_epa",
        "off_wp_pass_success_rate",
        "def_wp_pass_success_rate",
        "off_wp_rush_success_rate",
        "def_wp_rush_success_rate",
    )

    result = _opponent_adjusted_observations(raw, pregame)
    team_a = result.filter(pl.col("team_id") == "A").row(0, named=True)

    assert set(OPPONENT_ADJUSTED_METRICS).issubset(result.columns)
    assert team_a["off_opp_adj_pass_epa"] == pytest.approx(0.20)
    assert team_a["def_opp_adj_pass_epa"] == pytest.approx(-0.05)
    assert team_a["off_opp_adj_rush_epa"] == pytest.approx(0.15)


def test_shadow_ranking_uses_conservative_push_aware_ev_and_one_exposure() -> None:
    rows = [
        {
            "game_id": "g1",
            "market": "spreads",
            "selection": "HOME",
            "book": "pinnacle",
            "model_probability": 0.58,
            "market_fair_probability": 0.52,
            "push_probability": 0.05,
            "probability_margin": 0.02,
            "net_decimal_profit": 1.0,
        },
        {
            "game_id": "g1",
            "market": "spreads",
            "selection": "AWAY",
            "book": "retail",
            "model_probability": 0.56,
            "market_fair_probability": 0.50,
            "push_probability": 0.05,
            "probability_margin": 0.02,
            "net_decimal_profit": 1.0,
        },
        {
            "game_id": "g2",
            "market": "totals",
            "selection": "OVER",
            "book": "pinnacle",
            "model_probability": 0.53,
            "market_fair_probability": 0.52,
            "push_probability": 0.00,
            "probability_margin": 0.02,
            "net_decimal_profit": 1.0,
        },
    ]

    ranked = rank_shadow_candidates(rows, top=5)

    assert len(ranked) == 1
    assert ranked[0]["game_id"] == "g1"
    assert ranked[0]["selection"] == "HOME"
    assert ranked[0]["conservative_probability"] == pytest.approx(0.56)
    assert ranked[0]["conservative_probability_edge"] == pytest.approx(0.04)
    assert ranked[0]["conservative_expected_roi"] == pytest.approx(0.95 * 0.56 - 0.95 * 0.44)


def test_cli_exposes_challenger_weekly_workflow() -> None:
    result = CliRunner().invoke(app, ["challenger", "--help"])
    assert result.exit_code == 0
    for command in ("features", "train", "predict", "settle", "compare"):
        assert command in result.stdout

    predict_help = CliRunner().invoke(app, ["challenger", "predict", "--help"])
    assert predict_help.exit_code == 0
    assert "--latest" in predict_help.stdout


def test_latest_snapshot_resolver_uses_newest_complete_pregame_decision(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    with connect(settings) as connection:
        connection.execute(
            "INSERT INTO games(game_id,season,week,game_type,kickoff_utc,away_team,"
            "home_team,source,retrieved_at_utc,schema_version,content_hash) VALUES "
            "('game',2026,3,'REG','2026-09-27T17:00:00Z','AWAY','HOME','fixture',"
            "'2026-09-20T10:00:00Z','1','game-hash')"
        )
        for snapshot_id, purpose, retrieved in (
            ("older", "DECISION", "2026-09-20T10:00:00Z"),
            ("latest", "DECISION", "2026-09-20T11:00:00Z"),
            ("close", "CLOSE", "2026-09-20T12:00:00Z"),
        ):
            connection.execute(
                "INSERT INTO raw_snapshots(snapshot_id,provider,kind,snapshot_purpose,path,"
                "retrieved_at_utc,content_hash,byte_count) VALUES (?,?,?,?,?,?,?,1)",
                (
                    snapshot_id,
                    "fixture",
                    "full-board",
                    purpose,
                    f"{snapshot_id}.json",
                    retrieved,
                    snapshot_id,
                ),
            )
            connection.execute(
                "INSERT INTO api_requests(request_id,slot,request_kind,week_bucket,"
                "started_at_utc,completed_at_utc,status,raw_snapshot_id) VALUES "
                "(?,?,?,?,?,?, 'COMPLETE',?)",
                (
                    f"request-{snapshot_id}",
                    "manual",
                    "full-board",
                    "2026-09-22",
                    retrieved,
                    retrieved,
                    snapshot_id,
                ),
            )
            connection.execute(
                "INSERT INTO market_odds(snapshot_id,provider_event_id,game_id,"
                "commence_time_utc,bookmaker_key,bookmaker_title,bookmaker_group,market,"
                "selection,canonical_line,american_price,decimal_price,implied_probability,"
                "vig_free_probability,overround,source,retrieved_at_utc,schema_version,"
                "content_hash) VALUES (?,?,?,'2026-09-27T17:00:00Z','pinnacle','Pinnacle',"
                "'anchor','spreads','HOME',3.0,-110,1.909,0.5238,0.5,1.0476,'fixture',"
                "?,'1',?)",
                (snapshot_id, f"event-{snapshot_id}", "game", retrieved, f"quote-{snapshot_id}"),
            )
        connection.commit()

    assert (
        latest_valid_decision_snapshot(settings, as_of=datetime(2026, 9, 20, 13, 0, tzinfo=UTC))
        == "latest"
    )


def test_comparator_reports_missing_matched_evidence_without_inventing_rows(
    tmp_path,
) -> None:
    settings = Settings.for_root(tmp_path)
    with pytest.raises(ProspectiveDataError, match="sealed through Week 8"):
        compare_challenger(through_week=3, settings=settings)

    report = compare_challenger(through_week=8, settings=settings)
    assert report["status"] == "INSUFFICIENT_MATCHED_EVIDENCE"
    assert report["matched_evaluations"] == 0
    assert report["markets"]["spreads"]["challenger"] == {"eligible_non_push": 0}


def test_challenger_settlement_stays_sealed_until_week_eight_is_final(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    pl.DataFrame(
        {
            "season": [2026],
            "week": [8],
            "game_type": ["REG"],
            "kickoff_utc": ["2026-11-01T18:00:00Z"],
            "home_score": [None],
            "away_score": [None],
        }
    ).write_csv(tmp_path / "games.csv")

    with pytest.raises(ProspectiveDataError, match="sealed until every Week 8 game is final"):
        settle_challenger_predictions(datetime(2026, 11, 2, tzinfo=UTC), settings=settings)
