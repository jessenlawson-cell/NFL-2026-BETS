from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Final, cast

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import initialize_database
from nfl_bets.features.build import (
    HALF_LIFE_GRID,
    PRIOR_STRENGTH_GRID,
    build_lag_matrix,
    materialize_feature_configuration,
    validate_lag_boundaries,
)
from nfl_bets.features.challenger import CHALLENGER_METRICS
from nfl_bets.model.residuals import ProbabilityCalibrator, StratifiedResidualMapper
from nfl_bets.model.training import RIDGE_GRID, FeatureConfig, _append_history
from nfl_bets.util import atomic_write_text, iso_utc, sha256_bytes

MATCHUP_PAIRS: Final = {
    "wp_pass_epa": ("off_wp_pass_epa", "def_wp_pass_epa"),
    "wp_pass_success": ("off_wp_pass_success_rate", "def_wp_pass_success_rate"),
    "cpoe": ("off_cpoe", "def_cpoe"),
    "air_epa": ("off_air_epa", "def_air_epa"),
    "yac_epa": ("off_yac_epa", "def_yac_epa"),
    "pass_explosive": ("off_pass_explosive_rate", "def_pass_explosive_rate"),
    "wp_rush_epa": ("off_wp_rush_epa", "def_wp_rush_epa"),
    "wp_rush_success": ("off_wp_rush_success_rate", "def_wp_rush_success_rate"),
    "rush_explosive": ("off_rush_explosive_rate", "def_rush_explosive_rate"),
    "short_yard_success": (
        "off_short_yard_success_rate",
        "def_short_yard_success_rate",
    ),
    "sack_risk": ("off_sack_rate", "def_sack_rate"),
    "qb_hit_risk": ("off_qb_hit_rate", "def_qb_hit_rate"),
    "neutral_proe": ("off_neutral_proe", "def_neutral_proe_allowed"),
    "early_down_pass": (
        "off_early_down_pass_rate",
        "def_early_down_pass_rate_allowed",
    ),
    "no_huddle": ("off_no_huddle_rate", "def_no_huddle_rate_allowed"),
    "shotgun": ("off_shotgun_rate", "def_shotgun_rate_allowed"),
    "opp_adj_pass_epa": ("off_opp_adj_pass_epa", "def_opp_adj_pass_epa"),
    "opp_adj_rush_epa": ("off_opp_adj_rush_epa", "def_opp_adj_rush_epa"),
    "opp_adj_pass_success": (
        "off_opp_adj_pass_success",
        "def_opp_adj_pass_success",
    ),
    "opp_adj_rush_success": (
        "off_opp_adj_rush_success",
        "def_opp_adj_rush_success",
    ),
}
OPPONENT_BASE_PAIRS: Final = {
    "pass_epa": ("off_wp_pass_epa", "def_wp_pass_epa"),
    "rush_epa": ("off_wp_rush_epa", "def_wp_rush_epa"),
    "pass_success": ("off_wp_pass_success_rate", "def_wp_pass_success_rate"),
    "rush_success": ("off_wp_rush_success_rate", "def_wp_rush_success_rate"),
}
OPPONENT_ADJUSTED_METRICS: Final = tuple(
    f"{side}_opp_adj_{name}"
    for name in OPPONENT_BASE_PAIRS
    for side in ("off", "def")
)
MODEL_TEAM_METRICS: Final = (*CHALLENGER_METRICS, *OPPONENT_ADJUSTED_METRICS)
TEAM_METRICS: Final = (
    "st_epa",
    "st_success_rate",
    "st_field_goal_epa",
    "st_extra_point_epa",
    "st_punt_epa",
    "st_kickoff_epa",
    "off_snap_continuity",
    "def_snap_continuity",
)


def _pair_columns(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{name}_{suffix}" for name in names for suffix in ("difference", "sum"))


CHALLENGER_FEATURE_FAMILIES: Final = {
    "passing": _pair_columns(
        (
            "wp_pass_epa_matchup",
            "wp_pass_success_matchup",
            "cpoe_matchup",
            "air_epa_matchup",
            "yac_epa_matchup",
            "pass_explosive_matchup",
        )
    ),
    "rushing": _pair_columns(
        (
            "wp_rush_epa_matchup",
            "wp_rush_success_matchup",
            "rush_explosive_matchup",
            "short_yard_success_matchup",
        )
    ),
    "trenches": _pair_columns(("sack_risk_matchup", "qb_hit_risk_matchup")),
    "tendency": _pair_columns(
        ("neutral_proe_matchup", "early_down_pass_matchup", "no_huddle_matchup", "shotgun_matchup")
    ),
    "special_teams": _pair_columns(
        (
            "st_epa",
            "st_success_rate",
            "st_field_goal_epa",
            "st_extra_point_epa",
            "st_punt_epa",
            "st_kickoff_epa",
        )
    ),
    "continuity": _pair_columns(("off_snap_continuity", "def_snap_continuity")),
    "opponent_adjustment": _pair_columns(
        (
            "opp_adj_pass_epa_matchup",
            "opp_adj_rush_epa_matchup",
            "opp_adj_pass_success_matchup",
            "opp_adj_rush_success_matchup",
        )
    ),
    "context": ("rest_difference", "is_outdoors", "is_grass"),
}
CHALLENGER_FEATURE_COLUMNS: Final = tuple(
    column for columns in CHALLENGER_FEATURE_FAMILIES.values() for column in columns
)


@dataclass
class ChallengerCandidate:
    version: str
    feature_columns: tuple[str, ...]
    spread_model: Pipeline
    total_model: Pipeline
    spread_residuals: StratifiedResidualMapper
    total_residuals: StratifiedResidualMapper
    spread_calibrator: ProbabilityCalibrator
    total_calibrator: ProbabilityCalibrator
    spread_probability_margin: float
    total_probability_margin: float
    selected_half_life: float
    selected_prior_strength: float
    training_cutoff: str
    prospective_start_week: int
    feature_hash: str
    spec_hash: str


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _chronological_oof(
    matrix: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target: str,
    line_column: str,
    alpha: float,
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    seasons = sorted(int(value) for value in matrix["season"].unique().to_list())
    for validation_season in seasons[1:]:
        train = matrix.filter(pl.col("season") < validation_season)
        valid = matrix.filter(pl.col("season") == validation_season)
        if train.height < 30 or valid.is_empty():
            continue
        model = _pipeline(alpha).fit(
            train.select(feature_columns).to_numpy(), train[target].to_numpy()
        )
        rows.append(
            valid.select("game_id", "season", target, line_column).with_columns(
                pl.Series("projection", model.predict(valid.select(feature_columns).to_numpy()))
            )
        )
    if not rows:
        raise ValueError("Insufficient chronological challenger folds")
    return pl.concat(rows).rename({target: "actual", line_column: "line"})


def _probability_oof(
    matrix: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target: str,
    line_column: str,
    alpha: float,
    key_numbers: tuple[int, ...],
) -> pl.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    seasons = sorted(int(value) for value in matrix["season"].unique().to_list())
    for validation_season in seasons[1:]:
        train = matrix.filter(pl.col("season") < validation_season)
        valid = matrix.filter(pl.col("season") == validation_season)
        if train.height < 30 or valid.is_empty():
            continue
        model = _pipeline(alpha).fit(
            train.select(feature_columns).to_numpy(), train[target].to_numpy()
        )
        train_projection = model.predict(train.select(feature_columns).to_numpy())
        mapper = StratifiedResidualMapper(
            key_numbers=key_numbers,
            absolute_strata=bool(key_numbers),
        ).fit(train[target].to_numpy(), train_projection, train[line_column].to_numpy())
        valid_projection = model.predict(valid.select(feature_columns).to_numpy())
        for index, row in enumerate(
            valid.select("game_id", "season", target, line_column).iter_rows(named=True)
        ):
            line = float(row[line_column])
            actual = float(row[target])
            probability = mapper.probabilities(
                float(valid_projection[index]), line, line
            )
            non_push_mass = probability.win + probability.loss
            if non_push_mass <= 0.0:
                raise ValueError("OOF residual distribution has no non-push mass")
            rows.append(
                {
                    "game_id": str(row["game_id"]),
                    "season": int(row["season"]),
                    "actual": actual,
                    "line": line,
                    "projection": float(valid_projection[index]),
                    "raw_non_push_probability": probability.win / non_push_mass,
                    "push_probability": probability.push,
                    "is_push": math.isclose(actual, line, abs_tol=1e-9),
                    "outcome": float(actual > line),
                }
            )
    if not rows:
        raise ValueError("Insufficient chronological challenger probability folds")
    return pl.DataFrame(rows).sort(["season", "game_id"])


def _candidate_feature_sets() -> dict[str, tuple[str, ...]]:
    core_names = ("passing", "rushing", "trenches", "context")
    optional = ("tendency", "special_teams", "continuity", "opponent_adjustment")
    candidates: dict[str, tuple[str, ...]] = {}
    for count in range(len(optional) + 1):
        for selected in itertools.combinations(optional, count):
            names = (*core_names, *selected)
            candidates["+".join(names)] = tuple(
                column for name in names for column in CHALLENGER_FEATURE_FAMILIES[name]
            )
    return candidates


def _opponent_adjusted_observations(
    raw: pl.DataFrame, pregame: pl.DataFrame
) -> pl.DataFrame:
    required = {
        "game_id",
        "season",
        "week",
        "team_id",
        "opponent_team_id",
        *[metric for pair in OPPONENT_BASE_PAIRS.values() for metric in pair],
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"Opponent adjustment raw input missing columns: {sorted(missing)}")
    pregame_required = {
        "game_id",
        "team_id",
        *[metric for pair in OPPONENT_BASE_PAIRS.values() for metric in pair],
    }
    if missing := pregame_required - set(pregame.columns):
        raise ValueError(
            f"Opponent adjustment pregame input missing columns: {sorted(missing)}"
        )
    opponent = pregame.select(
        "game_id",
        pl.col("team_id").alias("opponent_team_id"),
        *[
            pl.col(metric).alias(f"opponent_{metric}")
            for pair in OPPONENT_BASE_PAIRS.values()
            for metric in pair
        ],
    )
    joined = raw.join(
        opponent,
        on=["game_id", "opponent_team_id"],
        how="left",
        validate="m:1",
    )
    expressions: list[pl.Expr] = []
    for name, (offense, defense) in OPPONENT_BASE_PAIRS.items():
        expressions.extend(
            [
                (pl.col(offense) - pl.col(f"opponent_{defense}")).alias(
                    f"off_opp_adj_{name}"
                ),
                (pl.col(defense) - pl.col(f"opponent_{offense}")).alias(
                    f"def_opp_adj_{name}"
                ),
            ]
        )
    return joined.with_columns(*expressions).select(
        "game_id", "season", "week", "team_id", *OPPONENT_ADJUSTED_METRICS
    )


def materialize_opponent_adjusted(
    lags: pl.DataFrame,
    base_materialized: pl.DataFrame,
    *,
    half_life: float,
    prior_strength: float,
) -> pl.DataFrame:
    observations = _opponent_adjusted_observations(lags, base_materialized)
    schedule_columns = [
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        "team_id",
        "opponent_team_id",
        "is_home",
    ]
    team_games = lags.select(schedule_columns).join(
        observations,
        on=["game_id", "season", "week", "team_id"],
        how="left",
        validate="1:1",
    )
    opponent_lags = build_lag_matrix(team_games, OPPONENT_ADJUSTED_METRICS)
    validate_lag_boundaries(opponent_lags)
    opponent_materialized = materialize_feature_configuration(
        opponent_lags,
        half_life=half_life,
        prior_strength=prior_strength,
        metrics=OPPONENT_ADJUSTED_METRICS,
    ).select("game_id", "team_id", *OPPONENT_ADJUSTED_METRICS)
    return base_materialized.join(
        opponent_materialized,
        on=["game_id", "team_id"],
        how="left",
        validate="1:1",
    )


def build_matchup_frame(team_metrics: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    required_team = {"game_id", "is_home", *MODEL_TEAM_METRICS}
    missing_team = required_team - set(team_metrics.columns)
    if missing_team:
        raise ValueError(f"Challenger team metrics missing columns: {sorted(missing_team)}")
    required_games = {"game_id", "home_rest", "away_rest", "roof", "surface"}
    missing_games = required_games - set(games.columns)
    if missing_games:
        raise ValueError(f"Challenger games input missing columns: {sorted(missing_games)}")

    team = team_metrics.with_columns(
        pl.col("off_short_yard_success_rate").fill_null(
            pl.col("off_wp_rush_success_rate")
        ),
        pl.col("def_short_yard_success_rate").fill_null(
            pl.col("def_wp_rush_success_rate")
        ),
        *[
            pl.col(metric).fill_null(pl.col("st_epa"))
            for metric in (
                "st_field_goal_epa",
                "st_extra_point_epa",
                "st_punt_epa",
                "st_kickoff_epa",
            )
        ],
    )
    home = team.filter(pl.col("is_home")).select(
        "game_id", *[pl.col(metric).alias(f"home_{metric}") for metric in MODEL_TEAM_METRICS]
    )
    away = team.filter(~pl.col("is_home")).select(
        "game_id", *[pl.col(metric).alias(f"away_{metric}") for metric in MODEL_TEAM_METRICS]
    )
    expressions: list[pl.Expr] = []
    for name, (offense, defense) in MATCHUP_PAIRS.items():
        home_matchup = pl.col(f"home_{offense}") + pl.col(f"away_{defense}")
        away_matchup = pl.col(f"away_{offense}") + pl.col(f"home_{defense}")
        expressions.extend(
            [
                (home_matchup - away_matchup).alias(f"{name}_matchup_difference"),
                (home_matchup + away_matchup).alias(f"{name}_matchup_sum"),
            ]
        )
    for metric in TEAM_METRICS:
        expressions.extend(
            [
                (pl.col(f"home_{metric}") - pl.col(f"away_{metric}")).alias(
                    f"{metric}_difference"
                ),
                (pl.col(f"home_{metric}") + pl.col(f"away_{metric}")).alias(
                    f"{metric}_sum"
                ),
            ]
        )
    roof = pl.col("roof").cast(pl.String).str.to_lowercase()
    surface = pl.col("surface").cast(pl.String).str.to_lowercase()
    return (
        games.join(home, on="game_id", how="inner", validate="1:1")
        .join(away, on="game_id", how="inner", validate="1:1")
        .with_columns(
            *expressions,
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_difference"),
            (roof == "outdoors").cast(pl.Float64).alias("is_outdoors"),
            surface.str.contains("grass").cast(pl.Float64).alias("is_grass"),
        )
    )


def assemble_challenger_matrix(
    team_metrics: pl.DataFrame,
    *,
    settings: Settings | None = None,
    through_week: int,
) -> pl.DataFrame:
    resolved = settings or get_settings()
    games = (
        pl.read_csv(resolved.root / "games.csv", infer_schema_length=100_000)
        .filter(
            (pl.col("season") >= 2021)
            & (pl.col("game_type") == "REG")
            & (
                (pl.col("season") < 2026)
                | ((pl.col("season") == 2026) & (pl.col("week") <= through_week))
            )
        )
        .drop_nulls(
            [
                "home_score",
                "away_score",
                "home_rest",
                "away_rest",
                "spread_line",
                "total_line",
                "roof",
                "surface",
            ]
        )
    )
    matrix = (
        build_matchup_frame(team_metrics, games)
        .with_columns(
            (pl.col("home_score") - pl.col("away_score")).alias("target_margin"),
            (pl.col("home_score") + pl.col("away_score")).alias("target_total"),
        )
        .drop_nulls([*CHALLENGER_FEATURE_COLUMNS, "target_margin", "target_total"])
        .sort(["season", "week", "kickoff_utc", "game_id"])
    )
    if matrix.is_empty():
        raise ValueError("Challenger training matrix is empty")
    return matrix


def _validate_training_cutoff(games: pl.DataFrame, *, through_week: int) -> None:
    cutoff = games.filter(
        (pl.col("season") == 2026)
        & (pl.col("week") == through_week)
        & (pl.col("game_type") == "REG")
    )
    if cutoff.is_empty():
        raise ValueError(f"No scheduled 2026 Week {through_week} games exist")
    completed = cutoff.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if completed.height != cutoff.height:
        raise ValueError(
            f"{completed.height} of {cutoff.height} Week {through_week} games are final; "
            "refusing challenger freeze"
        )


def score_challenger_contract(
    candidate: ChallengerCandidate,
    market: str,
    line: float,
    features: np.ndarray,
) -> dict[str, float]:
    if market == "spreads":
        model = candidate.spread_model
        mapper = candidate.spread_residuals
        calibrator = candidate.spread_calibrator
        probability_margin = candidate.spread_probability_margin
    elif market == "totals":
        model = candidate.total_model
        mapper = candidate.total_residuals
        calibrator = candidate.total_calibrator
        probability_margin = candidate.total_probability_margin
    else:
        raise ValueError(f"Unsupported challenger market: {market}")
    projection = float(model.predict(features)[0])
    empirical = mapper.probabilities(projection, line, line)
    non_push_mass = empirical.win + empirical.loss
    if non_push_mass <= 0.0:
        raise ValueError("Challenger empirical distribution has no non-push mass")
    raw_non_push = empirical.win / non_push_mass
    calibrated_non_push = float(calibrator.predict(np.asarray([raw_non_push]))[0])
    calibrated_non_push = float(np.clip(calibrated_non_push, 1e-8, 1.0 - 1e-8))
    model_win = (1.0 - empirical.push) * calibrated_non_push
    model_loss = (1.0 - empirical.push) * (1.0 - calibrated_non_push)
    if not math.isclose(model_win + empirical.push + model_loss, 1.0, abs_tol=1e-9):
        raise ValueError("Challenger outcome probabilities do not sum to one")
    return {
        "raw_football_projection": projection,
        "raw_adjustment": projection - line,
        "adjustment_weight": 1.0,
        "final_projection": projection,
        "raw_non_push_win_probability": float(raw_non_push),
        "calibrated_non_push_win_probability": calibrated_non_push,
        "model_win_probability": float(model_win),
        "model_push_probability": float(empirical.push),
        "model_loss_probability": float(model_loss),
        "probability_margin": float(probability_margin),
    }


def quantitative_drivers(
    candidate: ChallengerCandidate,
    market: str,
    features: np.ndarray,
    *,
    top: int = 3,
) -> list[dict[str, float | str]]:
    if top < 1:
        raise ValueError("top must be positive")
    if market == "spreads":
        model = candidate.spread_model
    elif market == "totals":
        model = candidate.total_model
    else:
        raise ValueError(f"Unsupported challenger market: {market}")
    scaled = np.asarray(model.named_steps["scale"].transform(features)[0], dtype=float)
    coefficients = np.asarray(model.named_steps["ridge"].coef_, dtype=float)
    contributions = scaled * coefficients
    ordered = sorted(
        zip(candidate.feature_columns, contributions, strict=True),
        key=lambda item: (-abs(float(item[1])), item[0]),
    )
    return [
        {"feature": feature, "projection_contribution": float(value)}
        for feature, value in ordered[:top]
    ]


def _oof_metrics(frame: pl.DataFrame) -> dict[str, float | int]:
    non_push = frame.filter(~pl.col("is_push"))
    if non_push.height < 30:
        raise ValueError("Challenger probability scoring requires 30 non-push rows")
    probability = np.clip(
        non_push["raw_non_push_probability"].to_numpy(), 1e-8, 1.0 - 1e-8
    )
    outcome = non_push["outcome"].to_numpy()
    log_loss = float(
        -np.mean(outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability))
    )
    brier = float(np.mean((probability - outcome) ** 2))
    rmse = float(root_mean_squared_error(frame["actual"], frame["projection"]))
    return {"log_loss": log_loss, "brier": brier, "rmse": rmse, "rows": frame.height}


def _development_hash(
    matrix: pl.DataFrame,
    feature_columns: tuple[str, ...],
    config: FeatureConfig,
    through_week: int,
) -> str:
    columns = [
        "game_id",
        "season",
        "week",
        *feature_columns,
        "target_margin",
        "target_total",
        "spread_line",
        "total_line",
    ]
    payload = matrix.select(columns).sort(["season", "week", "game_id"]).write_csv().encode()
    settings = (
        f"{config.half_life:.6f}|{config.prior_strength:.6f}|{through_week}|"
        + ",".join(feature_columns)
    ).encode()
    return sha256_bytes(settings + b"\n" + payload)


def _select_configuration(
    lags: pl.DataFrame,
    settings: Settings,
    through_week: int,
) -> tuple[
    FeatureConfig,
    pl.DataFrame,
    tuple[str, ...],
    dict[str, float],
    dict[str, pl.DataFrame],
    list[dict[str, Any]],
]:
    feature_sets = _candidate_feature_sets()
    evaluations: list[dict[str, Any]] = []
    choices: list[
        tuple[
            float,
            float,
            int,
            float,
            float,
            str,
            FeatureConfig,
            pl.DataFrame,
            tuple[str, ...],
            dict[str, float],
            dict[str, pl.DataFrame],
        ]
    ] = []
    for half_life in HALF_LIFE_GRID:
        for prior_strength in PRIOR_STRENGTH_GRID:
            config = FeatureConfig(half_life, prior_strength)
            base_materialized = materialize_feature_configuration(
                lags,
                half_life=half_life,
                prior_strength=prior_strength,
                metrics=CHALLENGER_METRICS,
            )
            materialized = materialize_opponent_adjusted(
                lags,
                base_materialized,
                half_life=half_life,
                prior_strength=prior_strength,
            )
            matrix = assemble_challenger_matrix(
                materialized, settings=settings, through_week=through_week
            )
            development = matrix.filter(pl.col("season") <= 2025)
            for feature_name, feature_columns in feature_sets.items():
                best: dict[str, tuple[dict[str, float | int], float, pl.DataFrame]] = {}
                for market, target, line, keys in (
                    ("spread", "target_margin", "spread_line", (3, 6, 7, 10, 14)),
                    ("total", "target_total", "total_line", ()),
                ):
                    options: list[
                        tuple[float, float, float, float, dict[str, float | int], pl.DataFrame]
                    ] = []
                    for alpha in RIDGE_GRID:
                        oof = _probability_oof(
                            development,
                            feature_columns=feature_columns,
                            target=target,
                            line_column=line,
                            alpha=alpha,
                            key_numbers=keys,
                        )
                        metrics = _oof_metrics(oof)
                        options.append(
                            (
                                float(metrics["log_loss"]),
                                float(metrics["brier"]),
                                float(metrics["rmse"]),
                                alpha,
                                metrics,
                                oof,
                            )
                        )
                    logloss, brier, rmse, alpha, metrics, oof = min(
                        options, key=lambda row: row[:4]
                    )
                    best[market] = (metrics, alpha, oof)
                    evaluations.append(
                        {
                            "half_life": half_life,
                            "prior_strength": prior_strength,
                            "feature_set": feature_name,
                            "market": market,
                            "ridge_alpha": alpha,
                            "log_loss": logloss,
                            "brier": brier,
                            "rmse": rmse,
                            "rows": metrics["rows"],
                        }
                    )
                combined_logloss = float(
                    np.mean([float(best[name][0]["log_loss"]) for name in ("spread", "total")])
                )
                combined_brier = float(
                    np.mean([float(best[name][0]["brier"]) for name in ("spread", "total")])
                )
                choices.append(
                    (
                        combined_logloss,
                        combined_brier,
                        len(feature_columns),
                        half_life,
                        prior_strength,
                        feature_name,
                        config,
                        matrix,
                        feature_columns,
                        {name: best[name][1] for name in ("spread", "total")},
                        {name: best[name][2] for name in ("spread", "total")},
                    )
                )
    if not choices:
        raise ValueError("No challenger configuration produced chronological predictions")
    selected = min(choices, key=lambda row: row[:6])
    return (
        selected[6],
        selected[7],
        selected[8],
        selected[9],
        selected[10],
        evaluations,
    )


def train_challenger_model(
    version: str,
    *,
    through_week: int = 2,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    if not version.startswith("challenger-"):
        raise ValueError("Challenger versions must start with 'challenger-'")
    if through_week < 1 or through_week >= 8:
        raise ValueError("The Week 3-8 challenger freeze requires through-week 1 through 7")
    artifact_path = resolved.artifacts_dir / "models" / version / "candidate.joblib"
    if artifact_path.parent.exists():
        raise FileExistsError(f"Model version {version} is already reserved")
    feature_path = resolved.runtime_dir / "features" / "challenger_team_game_lags.parquet"
    manifest_path = resolved.manifests_dir / "challenger_feature_inputs.latest.json"
    if not feature_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Build challenger feature inputs before training")
    feature_manifest = cast(
        dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if feature_manifest.get("content_hash") != sha256_bytes(feature_path.read_bytes()):
        raise RuntimeError("Challenger feature store hash does not match its manifest")
    if feature_manifest.get("through_week") != through_week:
        raise RuntimeError("Challenger feature manifest through-week does not match training")
    if feature_manifest.get("injury_model_status") != "DIAGNOSTIC_ONLY_UNWEIGHTED":
        raise RuntimeError("Unexpected challenger injury weighting policy")
    lags = pl.read_parquet(feature_path)
    validate_lag_boundaries(lags)
    _validate_training_cutoff(
        pl.read_csv(resolved.root / "games.csv", infer_schema_length=100_000),
        through_week=through_week,
    )
    config, matrix, features, alphas, oof, evaluations = _select_configuration(
        lags, resolved, through_week
    )
    x = matrix.select(features).to_numpy()
    spread_model = _pipeline(alphas["spread"]).fit(x, matrix["target_margin"].to_numpy())
    total_model = _pipeline(alphas["total"]).fit(x, matrix["target_total"].to_numpy())

    components: dict[str, dict[str, Any]] = {}
    for market, keys in (("spread", (3, 6, 7, 10, 14)), ("total", ())):
        frame = oof[market]
        mapper = StratifiedResidualMapper(
            key_numbers=keys, absolute_strata=bool(keys)
        ).fit(frame["actual"].to_numpy(), frame["projection"].to_numpy(), frame["line"].to_numpy())
        non_push = frame.filter(~pl.col("is_push"))
        calibrator = ProbabilityCalibrator().fit(
            non_push["raw_non_push_probability"].to_numpy(),
            non_push["outcome"].to_numpy().astype(int),
        )
        calibrated = calibrator.predict(non_push["raw_non_push_probability"].to_numpy())
        outcome = non_push["outcome"].to_numpy()
        brier = float(np.mean((calibrated - outcome) ** 2))
        margin = float(1.96 * math.sqrt(brier / non_push.height))
        components[market] = {
            "mapper": mapper,
            "calibrator": calibrator,
            "probability_margin": margin,
            "brier": brier,
            "log_loss": float(
                -np.mean(
                    outcome * np.log(np.clip(calibrated, 1e-8, 1.0 - 1e-8))
                    + (1.0 - outcome)
                    * np.log(np.clip(1.0 - calibrated, 1e-8, 1.0 - 1e-8))
                )
            ),
            "rows": non_push.height,
        }

    spec_path = resolved.root / "CHALLENGER_SPEC.md"
    if not spec_path.exists():
        raise FileNotFoundError("CHALLENGER_SPEC.md is required before freezing")
    spec_hash = sha256_bytes(spec_path.read_bytes())
    feature_hash = _development_hash(matrix, features, config, through_week)
    cutoff = f"2026-W{through_week:02d}"
    candidate = ChallengerCandidate(
        version=version,
        feature_columns=features,
        spread_model=spread_model,
        total_model=total_model,
        spread_residuals=components["spread"]["mapper"],
        total_residuals=components["total"]["mapper"],
        spread_calibrator=components["spread"]["calibrator"],
        total_calibrator=components["total"]["calibrator"],
        spread_probability_margin=components["spread"]["probability_margin"],
        total_probability_margin=components["total"]["probability_margin"],
        selected_half_life=config.half_life,
        selected_prior_strength=config.prior_strength,
        training_cutoff=cutoff,
        prospective_start_week=through_week + 1,
        feature_hash=feature_hash,
        spec_hash=spec_hash,
    )
    created_at = iso_utc()
    selected_feature_set = next(
        name for name, columns in _candidate_feature_sets().items() if columns == features
    )
    report = {
        "model_version": version,
        "status": "SHADOW_FROZEN_WEEK3_TO_8",
        "created_at_utc": created_at,
        "training_cutoff": cutoff,
        "prospective_start_week": through_week + 1,
        "prospective_end_week": 8,
        "training_rows": matrix.height,
        "selected_configuration": {
            "half_life": config.half_life,
            "prior_strength": config.prior_strength,
            "feature_set": selected_feature_set,
            "feature_columns": list(features),
            "spread_ridge_alpha": alphas["spread"],
            "total_ridge_alpha": alphas["total"],
        },
        "development_probability_metrics": {
            market: {
                "brier": components[market]["brier"],
                "log_loss": components[market]["log_loss"],
                "probability_margin_95": components[market]["probability_margin"],
                "non_push_rows": components[market]["rows"],
            }
            for market in ("spread", "total")
        },
        "all_family_evaluations": evaluations,
        "injury_model_status": "DIAGNOSTIC_ONLY_UNWEIGHTED",
        "market_input_status": "NOT_USED_FOR_POINT_PROJECTION",
        "feature_hash": feature_hash,
        "spec_hash": spec_hash,
    }
    report_path = resolved.reports_dir / f"model_{version}_development.json"
    atomic_write_text(report_path, json.dumps(report, sort_keys=True, indent=2))
    staging = resolved.artifacts_dir / "models" / f".staging-{version}-{uuid.uuid4()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        joblib.dump(candidate, staging / "candidate.joblib")
        metadata = {
            "version": version,
            "status": report["status"],
            "created_at_utc": created_at,
            "training_cutoff": cutoff,
            "prospective_start_week": through_week + 1,
            "prospective_end_week": 8,
            "training_rows": matrix.height,
            "selected_configuration": report["selected_configuration"],
            "feature_hash": feature_hash,
            "spec_hash": spec_hash,
            "report": str(report_path.relative_to(resolved.root)).replace("\\", "/"),
        }
        atomic_write_text(
            staging / "metadata.json", json.dumps(metadata, sort_keys=True, indent=2)
        )
        os.replace(staging, artifact_path.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _append_history(
        {
            "model_version": version,
            "created_at_utc": created_at,
            "command": "train-challenger",
            "development_end_season": 2026,
            "test_season": 2026,
            "spec_hash": spec_hash,
            "feature_hash": feature_hash,
            "artifact_path": str(artifact_path),
            "status": report["status"],
            "spread_alpha": 1.0,
            "total_alpha": 1.0,
            "spread_brier": components["spread"]["brier"],
            "spread_log_loss": components["spread"]["log_loss"],
            "spread_market_brier": None,
            "spread_market_log_loss": None,
            "total_brier": components["total"]["brier"],
            "total_log_loss": components["total"]["log_loss"],
            "total_market_brier": None,
            "total_market_log_loss": None,
            "notes": "Football-only shadow challenger frozen for Week 3 through Week 8; no stakes.",
            "source": "local:challenger-chronological-training",
            "retrieved_at_utc": created_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    manifest = {
        "model_version": version,
        "status": report["status"],
        "artifact": str(artifact_path.relative_to(resolved.root)).replace("\\", "/"),
        "artifact_hash": sha256_bytes(artifact_path.read_bytes()),
        "metadata_hash": sha256_bytes((artifact_path.parent / "metadata.json").read_bytes()),
        "feature_input_hash": feature_manifest["content_hash"],
        "development_feature_hash": feature_hash,
        "spec_hash": spec_hash,
        "training_cutoff": cutoff,
        "prospective_start_week": through_week + 1,
        "prospective_end_week": 8,
        "schema_version": resolved.schema_version,
    }
    atomic_write_text(
        resolved.manifests_dir / f"model_{version}_development.json",
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    freeze_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    policy = {
        "model_version": version,
        "status": report["status"],
        "allowed_decisions": ["PASS"],
        "artifact_hash": manifest["artifact_hash"],
        "metadata_hash": manifest["metadata_hash"],
        "spec_hash": spec_hash,
        "development_feature_hash": feature_hash,
        "frozen_feature_input_hash": feature_manifest["content_hash"],
        "freeze_commit": freeze_commit,
        "prospective_start_utc": created_at,
        "prospective_start_week": through_week + 1,
        "prospective_end_week": 8,
        "test_season": 2026,
        "formal_test_minimum_week": 8,
        "minimum_non_push_observations_per_market": 100,
        "canonical_bookmaker": "pinnacle",
        "quote_max_age_minutes": 30,
        "close_window_minutes_before_kickoff": {"earliest": 90, "latest": 5},
        "selection_rule_version": "challenger-0.1.0",
        "uncertainty_status": "MODELED_DEVELOPMENT_MARGIN_SHADOW_ONLY",
        "schema_version": resolved.schema_version,
    }
    atomic_write_text(
        resolved.manifests_dir / f"prospective_policy_{version}.json",
        json.dumps(policy, sort_keys=True, indent=2),
    )
    return cast(dict[str, Any], metadata)
