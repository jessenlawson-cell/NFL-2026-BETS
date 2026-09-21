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
from sklearn.ensemble import HistGradientBoostingRegressor
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
from nfl_bets.model.training import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    RIDGE_GRID,
    _append_history,
    _calibration,
    _market_weight,
    _reliability,
)
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
    f"{side}_opp_adj_{name}" for name in OPPONENT_BASE_PAIRS for side in ("off", "def")
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
MODEL_FAMILIES: Final = ("ridge", "tree", "blend_25", "blend_50", "blend_75")


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
class WeightedBlendRegressor:
    ridge_model: Any
    tree_model: Any
    ridge_weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.ridge_weight <= 1.0:
            raise ValueError("ridge_weight must be between zero and one")

    def predict(self, features: np.ndarray) -> np.ndarray:
        ridge = np.asarray(self.ridge_model.predict(features), dtype=float)
        tree = np.asarray(self.tree_model.predict(features), dtype=float)
        return self.ridge_weight * ridge + (1.0 - self.ridge_weight) * tree


@dataclass
class ChallengerMarketModel:
    feature_columns: tuple[str, ...]
    model_family: str
    model: Any
    ridge_model: Any
    tree_model: Any
    football_weight: float
    residuals: StratifiedResidualMapper
    calibrator: ProbabilityCalibrator
    probability_margin: float
    selected_half_life: float
    selected_prior_strength: float
    selected_feature_set: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray

    def __post_init__(self) -> None:
        if not 0.0 <= self.football_weight <= 1.0:
            raise ValueError("football_weight must be between zero and one")


@dataclass
class ChallengerCandidate:
    version: str
    spread: ChallengerMarketModel
    total: ChallengerMarketModel
    training_cutoff: str
    prospective_start_week: int
    feature_hash: str
    spec_hash: str


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _tree_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tree",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=10.0,
                    early_stopping=False,
                    random_state=20_260_920,
                ),
            )
        ]
    )


def _fit_regressors(
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float,
) -> tuple[Pipeline, Pipeline]:
    ridge = _pipeline(ridge_alpha).fit(features, target)
    tree = _tree_pipeline().fit(features, target)
    return ridge, tree


def _selected_regressor(
    family: str,
    ridge: Pipeline,
    tree: Pipeline,
) -> Any:
    if family == "ridge":
        return ridge
    if family == "tree":
        return tree
    if family.startswith("blend_"):
        ridge_weight = int(family.removeprefix("blend_")) / 100.0
        return WeightedBlendRegressor(ridge, tree, ridge_weight)
    raise ValueError(f"Unsupported challenger model family: {family}")


def _market_component(candidate: ChallengerCandidate, market: str) -> ChallengerMarketModel:
    if market == "spreads":
        return candidate.spread
    if market == "totals":
        return candidate.total
    raise ValueError(f"Unsupported challenger market: {market}")


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
    model_family: str = "ridge",
) -> pl.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    seasons = sorted(int(value) for value in matrix["season"].unique().to_list())
    for validation_season in seasons[1:]:
        train = matrix.filter(pl.col("season") < validation_season)
        valid = matrix.filter(pl.col("season") == validation_season)
        if train.height < 30 or valid.is_empty():
            continue
        train_x = train.select(feature_columns).to_numpy()
        valid_x = valid.select(feature_columns).to_numpy()
        train_target = train[target].to_numpy()
        if model_family == "ridge":
            ridge = _pipeline(alpha).fit(train_x, train_target)
            tree = ridge
        elif model_family == "tree":
            tree = _tree_pipeline().fit(train_x, train_target)
            ridge = tree
        else:
            ridge, tree = _fit_regressors(
                train_x,
                train_target,
                ridge_alpha=alpha,
            )
        model = _selected_regressor(model_family, ridge, tree)
        raw_train_projection = model.predict(train_x)
        raw_valid_projection = model.predict(valid_x)
        fold_weight = _market_weight(
            raw_train_projection,
            train[target].to_numpy(),
            train[line_column].to_numpy(),
        )
        train_projection = train[line_column].to_numpy() + fold_weight * (
            raw_train_projection - train[line_column].to_numpy()
        )
        mapper = StratifiedResidualMapper(
            key_numbers=key_numbers,
            absolute_strata=bool(key_numbers),
        ).fit(train[target].to_numpy(), train_projection, train[line_column].to_numpy())
        valid_projection = valid[line_column].to_numpy() + fold_weight * (
            raw_valid_projection - valid[line_column].to_numpy()
        )
        ridge_projection = ridge.predict(valid_x)
        tree_projection = tree.predict(valid_x)
        selected_columns = ["game_id", "season", target, line_column]
        if "market_probability" in valid.columns:
            selected_columns.append("market_probability")
        for index, row in enumerate(valid.select(selected_columns).iter_rows(named=True)):
            line = float(row[line_column])
            actual = float(row[target])
            probability = mapper.probabilities(float(valid_projection[index]), line, line)
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
                    "raw_projection": float(raw_valid_projection[index]),
                    "football_weight": fold_weight,
                    "ridge_projection": float(ridge_projection[index]),
                    "tree_projection": float(tree_projection[index]),
                    "raw_non_push_probability": probability.win / non_push_mass,
                    "push_probability": probability.push,
                    "is_push": math.isclose(actual, line, abs_tol=1e-9),
                    "outcome": float(actual > line),
                    "market_probability": float(row.get("market_probability", 0.5)),
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


def _opponent_adjusted_observations(raw: pl.DataFrame, pregame: pl.DataFrame) -> pl.DataFrame:
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
        raise ValueError(f"Opponent adjustment pregame input missing columns: {sorted(missing)}")
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
                (pl.col(offense) - pl.col(f"opponent_{defense}")).alias(f"off_opp_adj_{name}"),
                (pl.col(defense) - pl.col(f"opponent_{offense}")).alias(f"def_opp_adj_{name}"),
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


def _american_implied(column: str) -> pl.Expr:
    odds = pl.col(column).cast(pl.Float64)
    return pl.when(odds < 0.0).then((-odds) / ((-odds) + 100.0)).otherwise(100.0 / (odds + 100.0))


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
        pl.col("off_short_yard_success_rate").fill_null(pl.col("off_wp_rush_success_rate")),
        pl.col("def_short_yard_success_rate").fill_null(pl.col("def_wp_rush_success_rate")),
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
                (pl.col(f"home_{metric}") - pl.col(f"away_{metric}")).alias(f"{metric}_difference"),
                (pl.col(f"home_{metric}") + pl.col(f"away_{metric}")).alias(f"{metric}_sum"),
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
    matchup = build_matchup_frame(team_metrics, games)
    if {
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    }.issubset(matchup.columns):
        matchup = (
            matchup.drop_nulls(["home_spread_odds", "away_spread_odds", "over_odds", "under_odds"])
            .filter(
                (pl.col("home_spread_odds") != 0)
                & (pl.col("away_spread_odds") != 0)
                & (pl.col("over_odds") != 0)
                & (pl.col("under_odds") != 0)
            )
            .with_columns(
                (
                    _american_implied("home_spread_odds")
                    / (
                        _american_implied("home_spread_odds")
                        + _american_implied("away_spread_odds")
                    )
                ).alias("market_spread_probability"),
                (
                    _american_implied("over_odds")
                    / (_american_implied("over_odds") + _american_implied("under_odds"))
                ).alias("market_total_probability"),
            )
        )
    else:
        matchup = matchup.with_columns(
            pl.lit(0.5).alias("market_spread_probability"),
            pl.lit(0.5).alias("market_total_probability"),
        )
    matrix = (
        matchup.with_columns(
            (pl.col("home_score") - pl.col("away_score")).alias("target_margin"),
            (pl.col("home_score") + pl.col("away_score")).alias("target_total"),
        )
        .drop_nulls(
            [
                *CHALLENGER_FEATURE_COLUMNS,
                "target_margin",
                "target_total",
                "market_spread_probability",
                "market_total_probability",
            ]
        )
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
    *,
    market_probability: float | None = None,
) -> dict[str, float | int | str]:
    component = _market_component(candidate, market)
    projection = float(component.model.predict(features)[0])
    final_projection = line + component.football_weight * (projection - line)
    empirical = component.residuals.probabilities(final_projection, line, line)
    non_push_mass = empirical.win + empirical.loss
    if non_push_mass <= 0.0:
        raise ValueError("Challenger empirical distribution has no non-push mass")
    raw_non_push = empirical.win / non_push_mass
    calibrated_non_push = (
        float(market_probability)
        if component.football_weight == 0.0 and market_probability is not None
        else float(component.calibrator.predict(np.asarray([raw_non_push]))[0])
    )
    calibrated_non_push = float(np.clip(calibrated_non_push, 1e-8, 1.0 - 1e-8))
    model_win = (1.0 - empirical.push) * calibrated_non_push
    model_loss = (1.0 - empirical.push) * (1.0 - calibrated_non_push)
    if not math.isclose(model_win + empirical.push + model_loss, 1.0, abs_tol=1e-9):
        raise ValueError("Challenger outcome probabilities do not sum to one")
    ridge_projection = float(component.ridge_model.predict(features)[0])
    tree_projection = float(component.tree_model.predict(features)[0])
    scale = np.where(component.feature_std > 0.0, component.feature_std, 1.0)
    z_score = np.abs((features[0] - component.feature_mean) / scale)
    outside = (features[0] < component.feature_min) | (features[0] > component.feature_max)
    return {
        "raw_football_projection": projection,
        "raw_adjustment": projection - line,
        "adjustment_weight": component.football_weight,
        "final_projection": final_projection,
        "raw_non_push_win_probability": float(raw_non_push),
        "calibrated_non_push_win_probability": calibrated_non_push,
        "model_win_probability": float(model_win),
        "model_push_probability": float(empirical.push),
        "model_loss_probability": float(model_loss),
        "probability_margin": float(component.probability_margin),
        "selected_model_family": component.model_family,
        "signal_source": (
            "MARKET_CALIBRATION_ONLY" if component.football_weight == 0.0 else "FOOTBALL_ADJUSTED"
        ),
        "selected_feature_set": component.selected_feature_set,
        "component_disagreement": abs(ridge_projection - tree_projection),
        "feature_drift_max_abs_z": float(np.max(z_score)),
        "feature_outside_training_range": int(np.sum(outside)),
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
    component = _market_component(candidate, market)
    base = float(component.model.predict(features)[0])
    contributions = []
    for index in range(features.shape[1]):
        perturbed = features.copy()
        perturbed[0, index] = component.feature_mean[index]
        contributions.append(base - float(component.model.predict(perturbed)[0]))
    ordered = sorted(
        zip(component.feature_columns, contributions, strict=True),
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
    probability = np.clip(non_push["raw_non_push_probability"].to_numpy(), 1e-8, 1.0 - 1e-8)
    outcome = non_push["outcome"].to_numpy()
    log_loss = float(
        -np.mean(outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability))
    )
    brier = float(np.mean((probability - outcome) ** 2))
    rmse = float(root_mean_squared_error(frame["actual"], frame["projection"]))
    return {"log_loss": log_loss, "brier": brier, "rmse": rmse, "rows": frame.height}


def _probability_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return {
        "brier": float(np.mean(np.square(clipped - outcome))),
        "log_loss": float(
            -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))
        ),
    }


def _paired_probability_bootstrap(
    outcome: np.ndarray,
    model_probability: np.ndarray,
    market_probability: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences: dict[str, list[float]] = {"brier": [], "log_loss": []}
    size = len(outcome)
    for _ in range(BOOTSTRAP_SAMPLES):
        index = rng.integers(0, size, size=size)
        model = _probability_metrics(outcome[index], model_probability[index])
        market = _probability_metrics(outcome[index], market_probability[index])
        for name in differences:
            differences[name].append(model[name] - market[name])
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "model_minus_market": {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for name, values in differences.items()
        },
    }


def _edge_monotonicity(
    outcome: np.ndarray,
    model_probability: np.ndarray,
    market_probability: np.ndarray,
) -> dict[str, Any]:
    edge = model_probability - market_probability
    order = np.argsort(edge, kind="stable")
    bins = np.array_split(order, 5)
    rows = [
        {
            "bin": index,
            "count": int(len(selected)),
            "mean_edge": float(np.mean(edge[selected])),
            "observed_rate": float(np.mean(outcome[selected])),
        }
        for index, selected in enumerate(bins, start=1)
        if len(selected)
    ]
    observed = [float(row["observed_rate"]) for row in rows]
    return {
        "bins": rows,
        "observed_rate_nondecreasing": all(
            left <= right for left, right in zip(observed, observed[1:], strict=False)
        ),
    }


def _development_hash(selections: dict[str, dict[str, Any]], through_week: int) -> str:
    chunks: list[bytes] = []
    for market in ("spread", "total"):
        selected = selections[market]
        matrix = cast(pl.DataFrame, selected["matrix"])
        features = cast(tuple[str, ...], selected["feature_columns"])
        columns = [
            "game_id",
            "season",
            "week",
            *features,
            "target_margin",
            "target_total",
            "spread_line",
            "total_line",
        ]
        header = (
            f"{market}|{selected['half_life']:.6f}|{selected['prior_strength']:.6f}|"
            f"{selected['model_family']}|{selected['ridge_alpha']:.6f}|{through_week}|"
            + ",".join(features)
        ).encode()
        payload = matrix.select(columns).sort(["season", "week", "game_id"]).write_csv()
        chunks.append(header + b"\n" + payload.encode())
    return sha256_bytes(b"\n---\n".join(chunks))


def _select_configuration(
    lags: pl.DataFrame,
    settings: Settings,
    through_week: int,
    *,
    development_end_season: int = 2025,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    feature_sets = _candidate_feature_sets()
    evaluations: list[dict[str, Any]] = []
    choices: dict[str, list[tuple[Any, ...]]] = {"spread": [], "total": []}
    for half_life in HALF_LIFE_GRID:
        for prior_strength in PRIOR_STRENGTH_GRID:
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
            development = matrix.filter(pl.col("season") <= development_end_season)
            for feature_name, feature_columns in feature_sets.items():
                for market, target, line, probability_column, keys in (
                    (
                        "spread",
                        "target_margin",
                        "spread_line",
                        "market_spread_probability",
                        (3, 6, 7, 10, 14),
                    ),
                    (
                        "total",
                        "target_total",
                        "total_line",
                        "market_total_probability",
                        (),
                    ),
                ):
                    market_development = development.with_columns(
                        pl.col(probability_column).alias("market_probability")
                    )
                    options: list[
                        tuple[float, float, float, float, dict[str, float | int], pl.DataFrame]
                    ] = []
                    for alpha in RIDGE_GRID:
                        oof = _probability_oof(
                            market_development,
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
                    evaluations.append(
                        {
                            "half_life": half_life,
                            "prior_strength": prior_strength,
                            "feature_set": feature_name,
                            "market": market,
                            "model_family": "ridge",
                            "ridge_alpha": alpha,
                            "log_loss": logloss,
                            "brier": brier,
                            "rmse": rmse,
                            "rows": metrics["rows"],
                        }
                    )
                    choices[market].append(
                        (
                            logloss,
                            brier,
                            rmse,
                            len(feature_columns),
                            half_life,
                            prior_strength,
                            feature_name,
                            alpha,
                            matrix,
                            feature_columns,
                        )
                    )
    selections: dict[str, dict[str, Any]] = {}
    for market, target, line, probability_column, keys in (
        (
            "spread",
            "target_margin",
            "spread_line",
            "market_spread_probability",
            (3, 6, 7, 10, 14),
        ),
        (
            "total",
            "target_total",
            "total_line",
            "market_total_probability",
            (),
        ),
    ):
        if not choices[market]:
            raise ValueError(f"No challenger configuration produced {market} predictions")
        base = min(choices[market], key=lambda row: row[:8])
        matrix = cast(pl.DataFrame, base[8])
        feature_columns = cast(tuple[str, ...], base[9])
        development = matrix.filter(pl.col("season") <= development_end_season).with_columns(
            pl.col(probability_column).alias("market_probability")
        )
        family_options: list[tuple[Any, ...]] = []
        ridge_oof: pl.DataFrame | None = None
        for family in MODEL_FAMILIES:
            oof = _probability_oof(
                development,
                feature_columns=feature_columns,
                target=target,
                line_column=line,
                alpha=float(base[7]),
                key_numbers=keys,
                model_family=family,
            )
            metrics = _oof_metrics(oof)
            if family == "ridge":
                ridge_oof = oof
            family_options.append(
                (
                    float(metrics["log_loss"]),
                    float(metrics["brier"]),
                    float(metrics["rmse"]),
                    MODEL_FAMILIES.index(family),
                    family,
                    oof,
                )
            )
            evaluations.append(
                {
                    "half_life": base[4],
                    "prior_strength": base[5],
                    "feature_set": base[6],
                    "market": market,
                    "model_family": family,
                    "ridge_alpha": base[7],
                    **metrics,
                }
            )
        if ridge_oof is None:
            raise ValueError("Ridge baseline did not produce challenger predictions")
        market_rows = ridge_oof.filter(~pl.col("is_push"))
        market_metrics = _probability_metrics(
            market_rows["outcome"].to_numpy(),
            market_rows["market_probability"].to_numpy(),
        )
        market_rmse = float(
            root_mean_squared_error(ridge_oof["actual"].to_numpy(), ridge_oof["line"].to_numpy())
        )
        family_options.append(
            (
                market_metrics["log_loss"],
                market_metrics["brier"],
                market_rmse,
                -1,
                "market_only",
                ridge_oof,
            )
        )
        evaluations.append(
            {
                "half_life": base[4],
                "prior_strength": base[5],
                "feature_set": "market_only",
                "market": market,
                "model_family": "market_only",
                "ridge_alpha": base[7],
                "log_loss": market_metrics["log_loss"],
                "brier": market_metrics["brier"],
                "rmse": market_rmse,
                "rows": ridge_oof.height,
            }
        )
        selected_family = min(family_options, key=lambda row: row[:4])
        oof = cast(pl.DataFrame, selected_family[5])
        market_only = selected_family[4] == "market_only"
        football_weight = (
            0.0
            if market_only
            else _market_weight(
                oof["raw_projection"].to_numpy(),
                oof["actual"].to_numpy(),
                oof["line"].to_numpy(),
            )
        )
        selections[market] = {
            "matrix": matrix,
            "feature_columns": feature_columns,
            "feature_set": base[6],
            "half_life": float(base[4]),
            "prior_strength": float(base[5]),
            "ridge_alpha": float(base[7]),
            "model_family": "ridge" if market_only else selected_family[4],
            "selection_source": selected_family[4],
            "football_weight": football_weight,
            "oof": oof,
            "target": target,
            "line": line,
            "keys": keys,
            "market_probability_column": probability_column,
        }
    return selections, evaluations


def _fit_probability_contract(
    selection: dict[str, Any],
) -> tuple[StratifiedResidualMapper, ProbabilityCalibrator, pl.DataFrame, dict[str, Any]]:
    oof = cast(pl.DataFrame, selection["oof"])
    weight = float(selection["football_weight"])
    final_projection = oof["line"].to_numpy() + weight * (
        oof["raw_projection"].to_numpy() - oof["line"].to_numpy()
    )
    mapper = StratifiedResidualMapper(
        key_numbers=cast(tuple[int, ...], selection["keys"]),
        absolute_strata=bool(selection["keys"]),
    ).fit(oof["actual"].to_numpy(), final_projection, oof["line"].to_numpy())
    raw_probability: list[float] = []
    push_probability: list[float] = []
    for projection, line in zip(final_projection, oof["line"].to_numpy(), strict=True):
        probability = mapper.probabilities(float(projection), float(line), float(line))
        non_push_mass = probability.win + probability.loss
        if non_push_mass <= 0.0:
            raise ValueError("Challenger empirical distribution has no non-push mass")
        raw_probability.append(probability.win / non_push_mass)
        push_probability.append(probability.push)
    priced = oof.with_columns(
        pl.Series("final_projection", final_projection),
        pl.Series("contract_probability", raw_probability),
        pl.Series("contract_push_probability", push_probability),
    )
    non_push_rows = priced.filter(~pl.col("is_push"))
    calibrator = ProbabilityCalibrator().fit(
        non_push_rows["contract_probability"].to_numpy(),
        non_push_rows["outcome"].to_numpy().astype(int),
    )
    calibrated = calibrator.predict(non_push_rows["contract_probability"].to_numpy())
    outcome = non_push_rows["outcome"].to_numpy()
    market_probability = np.clip(non_push_rows["market_probability"].to_numpy(), 1e-8, 1.0 - 1e-8)
    if weight == 0.0:
        calibrated = market_probability
    model_metrics = _probability_metrics(outcome, calibrated)
    market_metrics = _probability_metrics(outcome, market_probability)
    evidence = {
        "model_probability": model_metrics,
        "market_probability": market_metrics,
        "calibration": _calibration(outcome, calibrated),
        "reliability": _reliability(outcome, calibrated),
        "paired_bootstrap": _paired_probability_bootstrap(outcome, calibrated, market_probability),
        "edge_monotonicity": _edge_monotonicity(outcome, calibrated, market_probability),
        "probability_margin_95": float(
            1.96 * math.sqrt(float(model_metrics["brier"]) / non_push_rows.height)
        ),
        "non_push_rows": non_push_rows.height,
        "projection_rmse": float(
            root_mean_squared_error(oof["actual"].to_numpy(), final_projection)
        ),
        "market_projection_rmse": float(
            root_mean_squared_error(oof["actual"].to_numpy(), oof["line"].to_numpy())
        ),
        "mean_component_disagreement": float(
            np.mean(np.abs(oof["ridge_projection"].to_numpy() - oof["tree_projection"].to_numpy()))
        ),
    }
    return mapper, calibrator, priced, evidence


def _fit_market_component(
    selection: dict[str, Any],
) -> tuple[ChallengerMarketModel, dict[str, Any]]:
    matrix = cast(pl.DataFrame, selection["matrix"])
    features = cast(tuple[str, ...], selection["feature_columns"])
    x = matrix.select(features).to_numpy()
    target = matrix[str(selection["target"])].to_numpy()
    ridge, tree = _fit_regressors(
        x,
        target,
        ridge_alpha=float(selection["ridge_alpha"]),
    )
    model = _selected_regressor(str(selection["model_family"]), ridge, tree)
    mapper, calibrator, _, evidence = _fit_probability_contract(selection)
    component = ChallengerMarketModel(
        feature_columns=features,
        model_family=str(selection["model_family"]),
        model=model,
        ridge_model=ridge,
        tree_model=tree,
        football_weight=float(selection["football_weight"]),
        residuals=mapper,
        calibrator=calibrator,
        probability_margin=float(evidence["probability_margin_95"]),
        selected_half_life=float(selection["half_life"]),
        selected_prior_strength=float(selection["prior_strength"]),
        selected_feature_set=str(selection["feature_set"]),
        feature_mean=np.mean(x, axis=0),
        feature_std=np.std(x, axis=0),
        feature_min=np.min(x, axis=0),
        feature_max=np.max(x, axis=0),
    )
    return component, evidence


def _score_outer_season(selection: dict[str, Any], validation_season: int) -> pl.DataFrame:
    matrix = cast(pl.DataFrame, selection["matrix"])
    features = cast(tuple[str, ...], selection["feature_columns"])
    target_name = str(selection["target"])
    line_name = str(selection["line"])
    probability_name = str(selection["market_probability_column"])
    train = matrix.filter(pl.col("season") < validation_season)
    valid = matrix.filter(pl.col("season") == validation_season)
    if train.height < 100 or valid.is_empty():
        raise ValueError("Insufficient rows for nested challenger validation")
    ridge, tree = _fit_regressors(
        train.select(features).to_numpy(),
        train[target_name].to_numpy(),
        ridge_alpha=float(selection["ridge_alpha"]),
    )
    model = _selected_regressor(str(selection["model_family"]), ridge, tree)
    mapper, calibrator, _, _ = _fit_probability_contract(selection)
    raw = model.predict(valid.select(features).to_numpy())
    final = valid[line_name].to_numpy() + float(selection["football_weight"]) * (
        raw - valid[line_name].to_numpy()
    )
    probabilities: list[float] = []
    pushes: list[float] = []
    for projection, line in zip(final, valid[line_name].to_numpy(), strict=True):
        probability = mapper.probabilities(float(projection), float(line), float(line))
        non_push = probability.win + probability.loss
        probabilities.append(probability.win / non_push)
        pushes.append(probability.push)
    calibrated = (
        valid[probability_name].to_numpy()
        if float(selection["football_weight"]) == 0.0
        else calibrator.predict(np.asarray(probabilities))
    )
    return (
        valid.select("game_id", "season", target_name, line_name, probability_name)
        .with_columns(
            pl.col(target_name).alias("actual"),
            pl.col(line_name).alias("line"),
            pl.col(probability_name).alias("market_probability"),
            pl.Series("projection", final),
            pl.Series("model_probability", calibrated),
            pl.Series("push_probability", pushes),
            (pl.col(target_name) == pl.col(line_name)).alias("is_push"),
            (pl.col(target_name) > pl.col(line_name)).cast(pl.Float64).alias("outcome"),
        )
        .select(
            "game_id",
            "season",
            "actual",
            "line",
            "market_probability",
            "projection",
            "model_probability",
            "push_probability",
            "is_push",
            "outcome",
        )
    )


def _nested_development_evidence(
    lags: pl.DataFrame,
    settings: Settings,
    through_week: int,
) -> dict[str, Any]:
    rows: dict[str, list[pl.DataFrame]] = {"spread": [], "total": []}
    folds: list[dict[str, Any]] = []
    for validation_season in (2024, 2025):
        selections, _ = _select_configuration(
            lags,
            settings,
            through_week,
            development_end_season=validation_season - 1,
        )
        for market in ("spread", "total"):
            fold = _score_outer_season(selections[market], validation_season)
            rows[market].append(fold)
            folds.append(
                {
                    "validation_season": validation_season,
                    "market": market,
                    "feature_set": selections[market]["feature_set"],
                    "model_family": selections[market]["model_family"],
                    "selection_source": selections[market]["selection_source"],
                    "football_weight": selections[market]["football_weight"],
                    "rows": fold.height,
                }
            )
    evidence: dict[str, Any] = {"folds": folds, "markets": {}}
    for market in ("spread", "total"):
        frame = pl.concat(rows[market]).sort(["season", "game_id"])
        non_push = frame.filter(~pl.col("is_push"))
        outcome = non_push["outcome"].to_numpy()
        model_probability = non_push["model_probability"].to_numpy()
        market_probability = non_push["market_probability"].to_numpy()
        evidence["markets"][market] = {
            "model_probability": _probability_metrics(outcome, model_probability),
            "market_probability": _probability_metrics(outcome, market_probability),
            "calibration": _calibration(outcome, model_probability),
            "paired_bootstrap": _paired_probability_bootstrap(
                outcome, model_probability, market_probability
            ),
            "edge_monotonicity": _edge_monotonicity(outcome, model_probability, market_probability),
            "rows": frame.height,
            "non_push_rows": non_push.height,
        }
    return evidence


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
    feature_manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
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
    nested_evidence = _nested_development_evidence(lags, resolved, through_week)
    selections, evaluations = _select_configuration(lags, resolved, through_week)
    spread, spread_evidence = _fit_market_component(selections["spread"])
    total, total_evidence = _fit_market_component(selections["total"])
    evidence = {"spread": spread_evidence, "total": total_evidence}

    spec_path = resolved.root / "CHALLENGER_SPEC.md"
    if not spec_path.exists():
        raise FileNotFoundError("CHALLENGER_SPEC.md is required before freezing")
    spec_hash = sha256_bytes(spec_path.read_bytes())
    feature_hash = _development_hash(selections, through_week)
    cutoff = f"2026-W{through_week:02d}"
    candidate = ChallengerCandidate(
        version=version,
        spread=spread,
        total=total,
        training_cutoff=cutoff,
        prospective_start_week=through_week + 1,
        feature_hash=feature_hash,
        spec_hash=spec_hash,
    )
    created_at = iso_utc()
    training_rows = {
        market: cast(pl.DataFrame, selections[market]["matrix"]).height
        for market in ("spread", "total")
    }
    selected_configuration = {
        market: {
            "half_life": selections[market]["half_life"],
            "prior_strength": selections[market]["prior_strength"],
            "feature_set": selections[market]["feature_set"],
            "feature_columns": list(selections[market]["feature_columns"]),
            "model_family": selections[market]["model_family"],
            "selection_source": selections[market]["selection_source"],
            "ridge_alpha": selections[market]["ridge_alpha"],
            "football_weight": selections[market]["football_weight"],
        }
        for market in ("spread", "total")
    }
    report = {
        "model_version": version,
        "status": "SHADOW_FROZEN_WEEK3_TO_8",
        "created_at_utc": created_at,
        "training_cutoff": cutoff,
        "prospective_start_week": through_week + 1,
        "prospective_end_week": 8,
        "training_rows": training_rows,
        "selected_configuration": selected_configuration,
        "development_probability_metrics": {
            market: evidence[market] for market in ("spread", "total")
        },
        "nested_chronological_evidence": nested_evidence,
        "all_family_evaluations": evaluations,
        "injury_model_status": "DIAGNOSTIC_ONLY_UNWEIGHTED",
        "market_input_status": "BENCHMARK_AND_CHRONOLOGICALLY_WEIGHTED_INPUT",
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
            "training_rows": training_rows,
            "selected_configuration": report["selected_configuration"],
            "feature_hash": feature_hash,
            "spec_hash": spec_hash,
            "report": str(report_path.relative_to(resolved.root)).replace("\\", "/"),
        }
        atomic_write_text(staging / "metadata.json", json.dumps(metadata, sort_keys=True, indent=2))
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
            "spread_alpha": selections["spread"]["football_weight"],
            "total_alpha": selections["total"]["football_weight"],
            "spread_brier": evidence["spread"]["model_probability"]["brier"],
            "spread_log_loss": evidence["spread"]["model_probability"]["log_loss"],
            "spread_market_brier": evidence["spread"]["market_probability"]["brier"],
            "spread_market_log_loss": evidence["spread"]["market_probability"]["log_loss"],
            "total_brier": evidence["total"]["model_probability"]["brier"],
            "total_log_loss": evidence["total"]["model_probability"]["log_loss"],
            "total_market_brier": evidence["total"]["market_probability"]["brier"],
            "total_market_log_loss": evidence["total"]["market_probability"]["log_loss"],
            "notes": "Frozen no-stakes Ridge/tree/blend shadow challenger for Weeks 3-8.",
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
        "selection_rule_version": version,
        "blind_until_week": 8,
        "uncertainty_status": "MODELED_DEVELOPMENT_MARGIN_SHADOW_ONLY",
        "schema_version": resolved.schema_version,
    }
    atomic_write_text(
        resolved.manifests_dir / f"prospective_policy_{version}.json",
        json.dumps(policy, sort_keys=True, indent=2),
    )
    return cast(dict[str, Any], metadata)
