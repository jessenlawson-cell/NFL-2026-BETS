from __future__ import annotations

import html
import json
import math
import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import initialize_database, transaction
from nfl_bets.features.build import (
    HALF_LIFE_GRID,
    METRICS,
    PRIOR_STRENGTH_GRID,
    materialize_feature_configuration,
    validate_lag_boundaries,
)
from nfl_bets.model.residuals import (
    KEY_NUMBERS,
    ProbabilityCalibrator,
    StratifiedResidualMapper,
)
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_text, canonical_hash, iso_utc, sha256_bytes

RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED = 20_260_913
DEVELOPMENT_END_SEASON = 2024
FEATURE_COLUMNS = tuple(
    [f"home_{metric}" for metric in METRICS]
    + [f"away_{metric}" for metric in METRICS]
    + ["rest_difference"]
)


@dataclass
class Candidate:
    version: str
    feature_columns: tuple[str, ...]
    spread_model: Pipeline
    total_model: Pipeline
    spread_ridge_alpha: float
    total_ridge_alpha: float
    spread_market_alpha: float
    total_market_alpha: float
    selected_half_life: float
    selected_prior_strength: float
    spread_residuals: StratifiedResidualMapper
    total_residuals: StratifiedResidualMapper
    spread_calibrator: ProbabilityCalibrator
    total_calibrator: ProbabilityCalibrator
    development_end_season: int
    training_rows: int
    feature_hash: str
    spec_hash: str


@dataclass(frozen=True)
class FeatureConfig:
    half_life: float
    prior_strength: float

    @property
    def key(self) -> str:
        return f"h{self.half_life:g}_p{self.prior_strength:g}"


def _market_fair_probability(first_price: float, second_price: float) -> float:
    def implied(price: float) -> float:
        if price == 0 or not math.isfinite(price):
            return math.nan
        return abs(price) / (abs(price) + 100.0) if price < 0 else 100.0 / (price + 100.0)

    first = implied(first_price)
    second = implied(second_price)
    total = first + second
    return first / total if total > 0 and math.isfinite(total) else math.nan


def assemble_game_matrix(
    settings: Settings | None = None,
    minimum_season: int = 2010,
    maximum_season: int | None = None,
    team_metrics: pl.DataFrame | None = None,
) -> pl.DataFrame:
    resolved = settings or get_settings()
    team = team_metrics if team_metrics is not None else pl.read_csv(
        resolved.root / "team_metrics.csv", infer_schema_length=100_000
    )
    games = pl.read_csv(resolved.root / "games.csv", infer_schema_length=100_000)
    games = games.filter(pl.col("season") >= minimum_season)
    team = team.filter(pl.col("season") >= minimum_season)
    if maximum_season is not None:
        games = games.filter(pl.col("season") <= maximum_season)
        team = team.filter(pl.col("season") <= maximum_season)
    games = games.drop_nulls(
        [
            "home_score",
            "away_score",
            "home_rest",
            "away_rest",
            "spread_line",
            "total_line",
            "home_spread_odds",
            "away_spread_odds",
            "over_odds",
            "under_odds",
        ]
    )
    home = team.filter(pl.col("is_home") == 1).select(
        "game_id", *[pl.col(metric).alias(f"home_{metric}") for metric in METRICS]
    )
    away = team.filter(pl.col("is_home") == 0).select(
        "game_id", *[pl.col(metric).alias(f"away_{metric}") for metric in METRICS]
    )
    matrix = (
        games.join(home, on="game_id", how="inner", validate="1:1")
        .join(away, on="game_id", how="inner", validate="1:1")
        .with_columns(
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_difference"),
            (pl.col("home_score") - pl.col("away_score")).alias("target_margin"),
            (pl.col("home_score") + pl.col("away_score")).alias("target_total"),
            pl.struct(["home_spread_odds", "away_spread_odds"])
            .map_elements(
                lambda row: _market_fair_probability(
                    row["home_spread_odds"], row["away_spread_odds"]
                ),
                return_dtype=pl.Float64,
            )
            .alias("market_spread_probability"),
            pl.struct(["over_odds", "under_odds"])
            .map_elements(
                lambda row: _market_fair_probability(row["over_odds"], row["under_odds"]),
                return_dtype=pl.Float64,
            )
            .alias("market_total_probability"),
        )
        .filter(pl.col("game_type") == "REG")
        .sort(["season", "week", "kickoff_utc", "game_id"])
    )
    required = [*FEATURE_COLUMNS, "target_margin", "target_total", "spread_line", "total_line"]
    return matrix.drop_nulls(required)


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _oof_predictions(
    matrix: pl.DataFrame, target: str, ridge_alpha: float, last_season: int
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    market_column = "spread_line" if target == "target_margin" else "total_line"
    for validation_season in range(2011, last_season + 1):
        train = matrix.filter(pl.col("season") < validation_season)
        valid = matrix.filter(pl.col("season") == validation_season)
        if train.height < 100 or valid.is_empty():
            continue
        model = _pipeline(ridge_alpha).fit(
            train.select(FEATURE_COLUMNS).to_numpy(), train[target].to_numpy()
        )
        prediction = model.predict(valid.select(FEATURE_COLUMNS).to_numpy())
        rows.append(
            valid.select("game_id", "season", target, market_column).with_columns(
                pl.Series("raw_projection", prediction)
            )
        )
    if not rows:
        raise ValueError("Insufficient chronological folds for training")
    return pl.concat(rows).rename({target: "actual", market_column: "market_projection"})


def _market_weight(raw: np.ndarray, actual: np.ndarray, market: np.ndarray) -> float:
    delta = raw - market
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(delta, actual - market) / denominator, 0.0, 1.0))


def _best_ridge(
    cache: dict[float, pl.DataFrame], before_season: int | None = None
) -> tuple[float, pl.DataFrame, float]:
    choices: list[tuple[float, float, pl.DataFrame]] = []
    for ridge, frame in cache.items():
        selected = (
            frame
            if before_season is None
            else frame.filter(pl.col("season") < before_season)
        )
        if selected.height < 100:
            continue
        rmse = float(
            root_mean_squared_error(
                selected["actual"].to_numpy(), selected["raw_projection"].to_numpy()
            )
        )
        choices.append((rmse, ridge, selected))
    if not choices:
        raise ValueError("Insufficient earlier out-of-fold rows for hyperparameter selection")
    rmse, ridge, frame = min(choices, key=lambda item: (item[0], item[1]))
    return ridge, frame, rmse


def _select_configuration(
    caches: dict[str, dict[str, dict[float, pl.DataFrame]]],
    configs: dict[str, FeatureConfig],
    before_season: int | None = None,
) -> tuple[FeatureConfig, dict[str, float], dict[str, pl.DataFrame]]:
    choices: list[
        tuple[float, float, float, FeatureConfig, dict[str, float], dict[str, pl.DataFrame]]
    ] = []
    for key, config in configs.items():
        ridge_values: dict[str, float] = {}
        selected_oof: dict[str, pl.DataFrame] = {}
        normalized_scores: list[float] = []
        try:
            for target in ("target_margin", "target_total"):
                ridge, frame, rmse = _best_ridge(caches[key][target], before_season)
                standard_deviation = max(
                    float(cast(float, frame["actual"].std())), 1e-9
                )
                ridge_values[target] = ridge
                selected_oof[target] = frame
                normalized_scores.append(rmse / standard_deviation)
        except ValueError:
            continue
        score = float(np.mean(normalized_scores))
        choices.append(
            (score, config.half_life, config.prior_strength, config, ridge_values, selected_oof)
        )
    if not choices:
        raise ValueError("No feature configuration has sufficient chronological development data")
    _, _, _, config, ridges, frames = min(choices, key=lambda item: item[:3])
    return config, ridges, frames


def _probability_from_mapper(
    mapper: StratifiedResidualMapper, projections: np.ndarray, lines: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            (lambda value: value.win / (value.win + value.loss))(
                mapper.probabilities(float(projection), float(line))
            )
            for projection, line in zip(projections, lines, strict=True)
        ],
        dtype=float,
    )


def _development_hash(matrix: pl.DataFrame, config: FeatureConfig) -> str:
    columns = [
        "game_id",
        "season",
        *FEATURE_COLUMNS,
        "target_margin",
        "target_total",
        "spread_line",
        "total_line",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    ]
    payload = matrix.select(columns).sort(["season", "game_id"]).write_csv().encode("utf-8")
    config_payload = f"{config.half_life:.6f}|{config.prior_strength:.6f}".encode("ascii")
    return sha256_bytes(config_payload + b"\n" + payload)


def _hash_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.exists() else "MISSING"


def _append_history(record: dict[str, Any], settings: Settings) -> None:
    record["content_hash"] = canonical_hash(record)
    columns = ARTIFACT_SCHEMAS["model_history"].columns
    with transaction(settings) as connection:
        placeholders = ",".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO model_history ({','.join(columns)}) VALUES ({placeholders})",
            tuple(record.get(column) for column in columns),
        )
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM model_history ORDER BY created_at_utc"
            )
        ]
    frame = pl.DataFrame(rows).select(columns) if rows else pl.DataFrame({c: [] for c in columns})
    atomic_write_text(settings.root / "model_history.csv", frame.write_csv())


def invalidate_candidate(
    version: str, reason: str, settings: Settings | None = None
) -> dict[str, Any]:
    """Permanently mark a frozen candidate unusable without rewriting its metadata."""
    resolved = settings or get_settings()
    artifact_dir = resolved.artifacts_dir / "models" / version
    metadata_path = artifact_dir / "metadata.json"
    invalidation_path = artifact_dir / "invalidation.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Frozen model {version} does not exist")
    if invalidation_path.exists():
        existing: dict[str, Any] = json.loads(
            invalidation_path.read_text(encoding="utf-8")
        )
        return existing
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    created_at = iso_utc()
    invalidation = {
        "model_version": version,
        "status": "INVALID_DATA_QUALITY",
        "invalidated_at_utc": created_at,
        "reason": reason,
        "untouched_test_consumed": False,
    }
    atomic_write_text(invalidation_path, json.dumps(invalidation, sort_keys=True, indent=2))
    selected = metadata.get("selected_configuration", {})
    _append_history(
        {
            "model_version": version,
            "created_at_utc": created_at,
            "command": "validate",
            "development_end_season": metadata.get("development_end_season"),
            "test_season": None,
            "spec_hash": metadata.get("spec_hash"),
            "feature_hash": metadata.get("feature_hash"),
            "artifact_path": str(artifact_dir / "candidate.joblib"),
            "status": "INVALID_DATA_QUALITY",
            "spread_alpha": selected.get("spread_market_alpha"),
            "total_alpha": selected.get("total_market_alpha"),
            "spread_brier": None,
            "spread_log_loss": None,
            "spread_market_brier": None,
            "spread_market_log_loss": None,
            "total_brier": None,
            "total_log_loss": None,
            "total_market_brier": None,
            "total_market_log_loss": None,
            "notes": reason,
            "source": "local:data-quality-validation",
            "retrieved_at_utc": created_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    return invalidation


def _materialize_authoritative_metrics(
    lags: pl.DataFrame,
    config: FeatureConfig,
    settings: Settings,
    as_of_utc: str,
) -> pl.DataFrame:
    matrix = materialize_feature_configuration(
        lags, half_life=config.half_life, prior_strength=config.prior_strength
    )
    retrieved_at = iso_utc()
    records: list[dict[str, Any]] = []
    for row in matrix.iter_rows(named=True):
        record: dict[str, Any] = {
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "kickoff_utc": iso_utc(row["kickoff_dt"]),
            "team_id": row["team_id"],
            "opponent_team_id": row["opponent_team_id"],
            "is_home": int(row["is_home"]),
            "games_available": row["games_available"],
            **{metric: row[metric] for metric in METRICS},
            "feature_as_of_utc": as_of_utc,
            "half_life": config.half_life,
            "source": f"nflverse:pbp-derived;prior_strength={config.prior_strength:g}",
            "retrieved_at_utc": retrieved_at,
            "source_updated_at_utc": None,
            "schema_version": settings.schema_version,
        }
        record["content_hash"] = canonical_hash(record)
        records.append(record)
    columns = ARTIFACT_SCHEMAS["team_metrics"].columns
    frame = pl.DataFrame(records).select(columns)
    atomic_write_text(settings.root / "team_metrics.csv", frame.write_csv())
    with transaction(settings) as connection:
        connection.execute("DELETE FROM team_metrics")
        placeholders = ",".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO team_metrics ({','.join(columns)}) VALUES ({placeholders})",
            [tuple(record.get(column) for column in columns) for record in records],
        )
    return matrix


def _projection_metrics(actual: np.ndarray, projection: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(actual, projection)),
        "mae": float(mean_absolute_error(actual, projection)),
    }


def _probability_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return {
        "brier": float(brier_score_loss(outcome, clipped)),
        "log_loss": float(log_loss(outcome, clipped, labels=[0, 1])),
    }


def _calibration(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    if np.unique(outcome).size < 2:
        return {"intercept": math.nan, "slope": math.nan}
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs").fit(logits, outcome)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def _reliability(outcome: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    bins = np.minimum((np.clip(probability, 0.0, 1.0) * 10).astype(int), 9)
    rows: list[dict[str, Any]] = []
    for index in range(10):
        mask = bins == index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": index + 1,
                "count": int(mask.sum()),
                "mean_probability": float(np.mean(probability[mask])),
                "observed_rate": float(np.mean(outcome[mask])),
            }
        )
    return rows


def _bootstrap(rows: pl.DataFrame) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seasons = np.asarray(sorted(rows["season"].unique().to_list()), dtype=int)
    samples: dict[str, list[float]] = {
        "spread_final_minus_market_rmse": [],
        "total_final_minus_market_rmse": [],
    }
    season_values = rows["season"].to_numpy()
    arrays = {
        column: rows[column].to_numpy()
        for column in (
            "target_margin",
            "target_total",
            "spread_final",
            "spread_line",
            "total_final",
            "total_line",
        )
    }
    by_season = {
        int(season): np.flatnonzero(season_values == season) for season in seasons
    }
    for _ in range(BOOTSTRAP_SAMPLES):
        selected: list[np.ndarray] = []
        for season in rng.choice(seasons, size=len(seasons), replace=True):
            indices = by_season[int(season)]
            selected.append(rng.choice(indices, size=len(indices), replace=True))
        sample_indices = np.concatenate(selected)
        for prefix, actual_column, final_column, market_column in (
            ("spread", "target_margin", "spread_final", "spread_line"),
            ("total", "target_total", "total_final", "total_line"),
        ):
            actual = arrays[actual_column][sample_indices]
            final = arrays[final_column][sample_indices]
            market = arrays[market_column][sample_indices]
            difference = root_mean_squared_error(
                actual, final
            ) - root_mean_squared_error(actual, market)
            samples[f"{prefix}_final_minus_market_rmse"].append(float(difference))
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def _coefficient_rows(candidate: Candidate) -> list[dict[str, Any]]:
    spread = candidate.spread_model.named_steps["ridge"].coef_
    total = candidate.total_model.named_steps["ridge"].coef_
    return [
        {
            "feature": feature,
            "spread_standardized": float(spread[index]),
            "total_standardized": float(total[index]),
        }
        for index, feature in enumerate(candidate.feature_columns)
    ]


def _write_development_reports(
    version: str,
    report: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, str]:
    prefix = settings.reports_dir / f"model_{version}_development"
    json_path = Path(f"{prefix}.json")
    csv_path = settings.reports_dir / f"model_{version}_development_folds.csv"
    coefficient_path = settings.reports_dir / f"model_{version}_coefficients.csv"
    html_path = Path(f"{prefix}.html")
    atomic_write_text(json_path, json.dumps(report, sort_keys=True, indent=2))
    atomic_write_text(csv_path, pl.DataFrame(fold_rows).write_csv())
    atomic_write_text(coefficient_path, pl.DataFrame(coefficient_rows).write_csv())
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
        for row in fold_rows
    )
    header_html = "".join(f"<th>{html.escape(key)}</th>" for key in fold_rows[0])
    selected_text = html.escape(json.dumps(report["selected_configuration"], indent=2))
    metrics_text = html.escape(json.dumps(report["development_metrics"], indent=2))
    document = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>NFL model {html.escape(version)} development</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#222}}
h1,h2{{color:#17365d}}
table{{border-collapse:collapse;width:100%}}
th{{background:#17365d;color:white}}
th,td{{padding:8px;border:1px solid #ccd6e0;text-align:left}}
pre{{background:#f4f7fa;padding:16px;overflow:auto}}
</style></head><body>
<h1>NFL Model {html.escape(version)} — Development Only</h1>
<p>Status: <strong>LOCKED_UNTESTED</strong>. The 2025 test has not been consumed.</p>
<h2>Selected configuration</h2><pre>{selected_text}</pre>
<h2>Development metrics</h2><pre>{metrics_text}</pre>
<h2>Chronological folds</h2><table><thead><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody></table>
</body></html>"""
    atomic_write_text(html_path, document)
    return {
        "json": str(json_path),
        "folds_csv": str(csv_path),
        "coefficients_csv": str(coefficient_path),
        "html": str(html_path),
    }


def _update_project_state(version: str, settings: Settings) -> None:
    path = settings.root / "PROJECT_STATE.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "- **Last Model Version:**": f"- **Last Model Version:** {version} (LOCKED_UNTESTED)",
        "- **System Status:**": (
            "- **System Status:** Production data synchronized, leakage-safe features "
            "materialized, and V1 candidate frozen; untouched 2025 test not executed"
        ),
        "- **Betting Status:**": (
            "- **Betting Status:** PASS-only; candidate is locked but untested"
        ),
    }
    updated = []
    for line in lines:
        replacement = next(
            (value for prefix, value in replacements.items() if line.startswith(prefix)),
            None,
        )
        updated.append(replacement or line)
    atomic_write_text(path, "\n".join(updated) + "\n")


def train_model(version: str, settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    artifact_path = resolved.artifacts_dir / "models" / version / "candidate.joblib"
    if artifact_path.exists() or artifact_path.parent.exists():
        raise FileExistsError(f"Model version {version} is already frozen or partially reserved")

    lag_path = resolved.runtime_dir / "features" / "team_game_lags.parquet"
    configs: dict[str, FeatureConfig] = {}
    feature_frames: dict[str, pl.DataFrame] = {}
    if lag_path.exists():
        lags = pl.read_parquet(lag_path)
        validate_lag_boundaries(lags)
        for half_life in HALF_LIFE_GRID:
            for prior_strength in PRIOR_STRENGTH_GRID:
                config = FeatureConfig(half_life, prior_strength)
                configs[config.key] = config
                feature_frames[config.key] = materialize_feature_configuration(
                    lags, half_life=half_life, prior_strength=prior_strength
                )
    else:
        # Deterministic fixture compatibility; production acceptance requires the lag store.
        config = FeatureConfig(2.0, 1.0)
        configs[config.key] = config
        feature_frames[config.key] = pl.read_csv(
            resolved.root / "team_metrics.csv", infer_schema_length=100_000
        )

    matrices: dict[str, pl.DataFrame] = {}
    caches: dict[str, dict[str, dict[float, pl.DataFrame]]] = {}
    for key, team_frame in feature_frames.items():
        matrix = assemble_game_matrix(
            resolved,
            minimum_season=2010,
            maximum_season=DEVELOPMENT_END_SEASON,
            team_metrics=team_frame,
        )
        maximum_season = cast(int | None, matrix["season"].max())
        if matrix.is_empty() or maximum_season != DEVELOPMENT_END_SEASON:
            raise ValueError("Development data must include the 2024 regular season")
        if maximum_season is not None and maximum_season >= 2025:
            raise RuntimeError("Training boundary violation: 2025 data entered development")
        matrices[key] = matrix
        caches[key] = {"target_margin": {}, "target_total": {}}
        for target in ("target_margin", "target_total"):
            for ridge in RIDGE_GRID:
                caches[key][target][ridge] = _oof_predictions(
                    matrix, target, ridge, DEVELOPMENT_END_SEASON
                )

    nested_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for season in range(2012, DEVELOPMENT_END_SEASON + 1):
        try:
            selected_config, selected_ridges, selected_inner = _select_configuration(
                caches, configs, before_season=season
            )
        except ValueError:
            continue
        matrix = matrices[selected_config.key]
        train = matrix.filter(pl.col("season") < season)
        valid = matrix.filter(pl.col("season") == season)
        if train.height < 100 or valid.is_empty():
            continue
        fold_payload: dict[str, Any] = {
            "season": season,
            "rows": valid.height,
            "half_life": selected_config.half_life,
            "prior_strength": selected_config.prior_strength,
        }
        output: dict[str, np.ndarray] = {}
        for prefix, target, line_column in (
            ("spread", "target_margin", "spread_line"),
            ("total", "target_total", "total_line"),
        ):
            ridge = selected_ridges[target]
            model = _pipeline(ridge).fit(
                train.select(FEATURE_COLUMNS).to_numpy(), train[target].to_numpy()
            )
            raw = model.predict(valid.select(FEATURE_COLUMNS).to_numpy())
            inner = selected_inner[target]
            market_alpha = _market_weight(
                inner["raw_projection"].to_numpy(),
                inner["actual"].to_numpy(),
                inner["market_projection"].to_numpy(),
            )
            line = valid[line_column].to_numpy()
            final = market_alpha * raw + (1.0 - market_alpha) * line
            inner_final = (
                market_alpha * inner["raw_projection"].to_numpy()
                + (1.0 - market_alpha) * inner["market_projection"].to_numpy()
            )
            mapper = StratifiedResidualMapper(
                key_numbers=KEY_NUMBERS if prefix == "spread" else (),
                absolute_strata=prefix == "spread",
            ).fit(inner["actual"].to_numpy(), inner_final, inner["market_projection"].to_numpy())
            probability = _probability_from_mapper(mapper, final, line)
            output[f"{prefix}_raw"] = raw
            output[f"{prefix}_final"] = final
            output[f"{prefix}_probability"] = probability
            fold_payload[f"{prefix}_ridge_alpha"] = ridge
            fold_payload[f"{prefix}_market_alpha"] = market_alpha
            fold_payload[f"{prefix}_raw_rmse"] = float(
                root_mean_squared_error(valid[target].to_numpy(), raw)
            )
            fold_payload[f"{prefix}_final_rmse"] = float(
                root_mean_squared_error(valid[target].to_numpy(), final)
            )
            fold_payload[f"{prefix}_market_rmse"] = float(
                root_mean_squared_error(valid[target].to_numpy(), line)
            )
        fold_rows.append(fold_payload)
        for index, row in enumerate(valid.iter_rows(named=True)):
            nested_rows.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "target_margin": row["target_margin"],
                    "target_total": row["target_total"],
                    "spread_line": row["spread_line"],
                    "total_line": row["total_line"],
                    "market_spread_probability": row["market_spread_probability"],
                    "market_total_probability": row["market_total_probability"],
                    **{name: values[index] for name, values in output.items()},
                }
            )
    if not nested_rows:
        raise ValueError("No nested chronological development predictions were produced")
    nested = pl.DataFrame(nested_rows).sort(["season", "game_id"])

    selected_config, selected_ridges, selected_oof = _select_configuration(caches, configs)
    selected_matrix = matrices[selected_config.key]
    final_components: dict[str, Any] = {}
    for prefix, target in (("spread", "target_margin"), ("total", "target_total")):
        oof = selected_oof[target]
        alpha = _market_weight(
            oof["raw_projection"].to_numpy(),
            oof["actual"].to_numpy(),
            oof["market_projection"].to_numpy(),
        )
        final = alpha * oof["raw_projection"].to_numpy() + (1.0 - alpha) * oof[
            "market_projection"
        ].to_numpy()
        mapper = StratifiedResidualMapper(
            key_numbers=KEY_NUMBERS if prefix == "spread" else (),
            absolute_strata=prefix == "spread",
        ).fit(oof["actual"].to_numpy(), final, oof["market_projection"].to_numpy())
        probability = _probability_from_mapper(mapper, final, oof["market_projection"].to_numpy())
        non_push = oof["actual"].to_numpy() != oof["market_projection"].to_numpy()
        outcomes = (
            oof["actual"].to_numpy()[non_push] > oof["market_projection"].to_numpy()[non_push]
        ).astype(int)
        calibrator = ProbabilityCalibrator().fit(probability[non_push], outcomes)
        final_components[prefix] = {
            "alpha": alpha,
            "mapper": mapper,
            "calibrator": calibrator,
            "oof": oof,
            "probability": probability,
            "outcomes": outcomes,
            "non_push": non_push,
        }

    if lag_path.exists():
        feature_manifest = json.loads(
            (resolved.manifests_dir / "team_feature_inputs.latest.json").read_text(encoding="utf-8")
        )
        selected_team = _materialize_authoritative_metrics(
            lags,
            selected_config,
            resolved,
            str(feature_manifest["as_of_utc"]),
        )
        selected_matrix = assemble_game_matrix(
            resolved,
            minimum_season=2010,
            maximum_season=DEVELOPMENT_END_SEASON,
            team_metrics=selected_team,
        )
        # Production candidates cannot freeze until the selected authoritative matrix passes.
        from nfl_bets.validation import validate_all

        validate_all(resolved)

    feature_hash = _development_hash(selected_matrix, selected_config)
    spec_hash = _hash_file(resolved.root / "MODEL_SPEC.md")
    x = selected_matrix.select(FEATURE_COLUMNS).to_numpy()
    spread_model = _pipeline(selected_ridges["target_margin"]).fit(
        x, selected_matrix["target_margin"].to_numpy()
    )
    total_model = _pipeline(selected_ridges["target_total"]).fit(
        x, selected_matrix["target_total"].to_numpy()
    )
    candidate = Candidate(
        version=version,
        feature_columns=FEATURE_COLUMNS,
        spread_model=spread_model,
        total_model=total_model,
        spread_ridge_alpha=selected_ridges["target_margin"],
        total_ridge_alpha=selected_ridges["target_total"],
        spread_market_alpha=final_components["spread"]["alpha"],
        total_market_alpha=final_components["total"]["alpha"],
        selected_half_life=selected_config.half_life,
        selected_prior_strength=selected_config.prior_strength,
        spread_residuals=final_components["spread"]["mapper"],
        total_residuals=final_components["total"]["mapper"],
        spread_calibrator=final_components["spread"]["calibrator"],
        total_calibrator=final_components["total"]["calibrator"],
        development_end_season=DEVELOPMENT_END_SEASON,
        training_rows=selected_matrix.height,
        feature_hash=feature_hash,
        spec_hash=spec_hash,
    )

    development_metrics: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    for prefix, actual_column, raw_column, final_column, line_column, market_probability_column in (
        (
            "spread",
            "target_margin",
            "spread_raw",
            "spread_final",
            "spread_line",
            "market_spread_probability",
        ),
        (
            "total",
            "target_total",
            "total_raw",
            "total_final",
            "total_line",
            "market_total_probability",
        ),
    ):
        actual = nested[actual_column].to_numpy()
        line = nested[line_column].to_numpy()
        raw = nested[raw_column].to_numpy()
        final = nested[final_column].to_numpy()
        uncalibrated = nested[f"{prefix}_probability"].to_numpy()
        non_push = actual != line
        outcome = (actual[non_push] > line[non_push]).astype(int)
        calibration = ProbabilityCalibrator().fit(uncalibrated[non_push], outcome)
        calibrated = calibration.predict(uncalibrated[non_push])
        market_probability = nested[market_probability_column].to_numpy()[non_push]
        development_metrics[prefix] = {
            "raw": _projection_metrics(actual, raw),
            "market": _projection_metrics(actual, line),
            "market_regressed": _projection_metrics(actual, final),
            "probability": _probability_metrics(outcome, calibrated),
            "market_probability": _probability_metrics(outcome, market_probability),
            "calibration": _calibration(outcome, calibrated),
        }
        reliability[prefix] = _reliability(outcome, calibrated)

    coefficient_rows = _coefficient_rows(candidate)
    report: dict[str, Any] = {
        "model_version": version,
        "status": "LOCKED_UNTESTED",
        "development_end_season": DEVELOPMENT_END_SEASON,
        "untouched_test_season": 2025,
        "untouched_test_consumed": False,
        "training_rows": selected_matrix.height,
        "selected_configuration": {
            "half_life": selected_config.half_life,
            "prior_strength": selected_config.prior_strength,
            "spread_ridge_alpha": selected_ridges["target_margin"],
            "total_ridge_alpha": selected_ridges["target_total"],
            "spread_market_alpha": candidate.spread_market_alpha,
            "total_market_alpha": candidate.total_market_alpha,
        },
        "development_metrics": development_metrics,
        "reliability": reliability,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "intervals": _bootstrap(nested),
        },
        "feature_hash": feature_hash,
        "spec_hash": spec_hash,
        "folds": fold_rows,
        "coefficients": coefficient_rows,
    }
    report_paths = _write_development_reports(
        version, report, fold_rows, coefficient_rows, resolved
    )

    staging = resolved.artifacts_dir / "models" / f".staging-{version}-{uuid.uuid4()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        joblib.dump(candidate, staging / "candidate.joblib")
        metadata = {
            "version": version,
            "status": "LOCKED_UNTESTED",
            "created_at_utc": iso_utc(),
            "development_end_season": DEVELOPMENT_END_SEASON,
            "untouched_test_season": 2025,
            "untouched_test_consumed": False,
            "training_rows": selected_matrix.height,
            "selected_configuration": report["selected_configuration"],
            "feature_hash": feature_hash,
            "spec_hash": spec_hash,
            "reports": report_paths,
        }
        atomic_write_text(staging / "metadata.json", json.dumps(metadata, sort_keys=True, indent=2))
        os.replace(staging, artifact_path.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    created_at = iso_utc()
    _append_history(
        {
            "model_version": version,
            "created_at_utc": created_at,
            "command": "train",
            "development_end_season": DEVELOPMENT_END_SEASON,
            "test_season": None,
            "spec_hash": spec_hash,
            "feature_hash": feature_hash,
            "artifact_path": str(artifact_path),
            "status": "LOCKED_UNTESTED",
            "spread_alpha": candidate.spread_market_alpha,
            "total_alpha": candidate.total_market_alpha,
            "spread_brier": None,
            "spread_log_loss": None,
            "spread_market_brier": None,
            "spread_market_log_loss": None,
            "total_brier": None,
            "total_log_loss": None,
            "total_market_brier": None,
            "total_market_log_loss": None,
            "notes": "Nested chronological development complete; untouched 2025 test not consumed.",
            "source": "local:chronological-training",
            "retrieved_at_utc": created_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    manifest = {
        "model_version": version,
        "status": "LOCKED_UNTESTED",
        "artifact": str(artifact_path.relative_to(resolved.root)).replace("\\", "/"),
        "artifact_hash": sha256_bytes(artifact_path.read_bytes()),
        "metadata_hash": sha256_bytes((artifact_path.parent / "metadata.json").read_bytes()),
        "feature_hash": feature_hash,
        "spec_hash": spec_hash,
        "untouched_test_consumed": False,
        "created_at_utc": created_at,
        "schema_version": resolved.schema_version,
    }
    atomic_write_text(
        resolved.manifests_dir / f"model_{version}_development.json",
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    _update_project_state(version, resolved)
    return metadata


def _calibration_is_acceptable(values: dict[str, float]) -> bool:
    return abs(values["intercept"]) <= 0.20 and 0.50 <= values["slope"] <= 1.50


def _test_input_hash(matrix: pl.DataFrame, candidate: Candidate) -> str:
    columns = [
        "game_id",
        "season",
        "week",
        *candidate.feature_columns,
        "target_margin",
        "target_total",
        "spread_line",
        "total_line",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    ]
    payload = matrix.select(columns).sort(["season", "week", "game_id"]).write_csv()
    return sha256_bytes(payload.encode("utf-8"))


def _test_gate_results(
    spread_candidate: dict[str, float],
    spread_market: dict[str, float],
    spread_calibration: dict[str, float],
    total_candidate: dict[str, float],
    total_market: dict[str, float],
    total_calibration: dict[str, float],
) -> dict[str, bool]:
    return {
        "spread_brier_beats_market": spread_candidate["brier"] < spread_market["brier"],
        "spread_log_loss_beats_market": (
            spread_candidate["log_loss"] < spread_market["log_loss"]
        ),
        "spread_calibration_acceptable": _calibration_is_acceptable(spread_calibration),
        "total_brier_beats_market": total_candidate["brier"] < total_market["brier"],
        "total_log_loss_beats_market": (
            total_candidate["log_loss"] < total_market["log_loss"]
        ),
        "total_calibration_acceptable": _calibration_is_acceptable(total_calibration),
    }


def _write_test_reports(
    version: str,
    season: int,
    report: dict[str, Any],
    audit: pl.DataFrame,
    settings: Settings,
) -> dict[str, str]:
    prefix = settings.reports_dir / f"model_{version}_test_{season}"
    json_path = Path(f"{prefix}.json")
    audit_path = settings.reports_dir / f"model_{version}_test_{season}_games.csv"
    gates_path = settings.reports_dir / f"model_{version}_test_{season}_gates.csv"
    html_path = Path(f"{prefix}.html")
    paths = (json_path, audit_path, gates_path, html_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Untouched-test report path already exists: {existing}")

    gate_rows = [
        {"gate": gate, "passed": passed}
        for gate, passed in cast(dict[str, bool], report["promotion_gates"]).items()
    ]
    status = html.escape(str(report["status"]))
    gate_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['gate']))}</td>"
        f"<td>{'PASS' if row['passed'] else 'FAIL'}</td>"
        "</tr>"
        for row in gate_rows
    )
    metrics_text = html.escape(
        json.dumps({"spread": report["spread"], "total": report["total"]}, indent=2)
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>NFL model {html.escape(version)} untouched {season} test</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#222}}
h1,h2{{color:#17365d}}
table{{border-collapse:collapse;width:100%}}
th{{background:#17365d;color:white}}
th,td{{padding:8px;border:1px solid #ccd6e0;text-align:left}}
pre{{background:#f4f7fa;padding:16px;overflow:auto}}
</style></head><body>
<h1>NFL Model {html.escape(version)} — Untouched {season} Test</h1>
<p>Final status: <strong>{status}</strong>. This test is permanently consumed.</p>
<p>No test result may be used to retune V1.</p>
<h2>Promotion gates</h2>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_html}</tbody></table>
<h2>Scoring and calibration</h2><pre>{metrics_text}</pre>
</body></html>"""

    atomic_write_text(json_path, json.dumps(report, sort_keys=True, indent=2))
    atomic_write_text(audit_path, audit.write_csv())
    atomic_write_text(gates_path, pl.DataFrame(gate_rows).write_csv())
    atomic_write_text(html_path, document)
    return {
        "json": str(json_path),
        "games_csv": str(audit_path),
        "gates_csv": str(gates_path),
        "html": str(html_path),
    }


def _update_test_project_state(version: str, status: str, settings: Settings) -> None:
    path = settings.root / "PROJECT_STATE.md"
    if not path.exists():
        return
    promoted = status == "PROMOTED"
    replacements = {
        "- **Last Model Version:**": f"- **Last Model Version:** {version} ({status})",
        "- **System Status:**": (
            f"- **System Status:** Untouched 2025 evaluation consumed; model {version} "
            + ("passed every promotion gate" if promoted else "failed one or more promotion gates")
        ),
        "- **Betting Status:**": (
            "- **Betting Status:** Paper-trading candidate eligible; no live market evaluated"
            if promoted
            else "- **Betting Status:** PASS-only; V1 candidate was not promoted"
        ),
    }
    updated = []
    for line in path.read_text(encoding="utf-8").splitlines():
        replacement = next(
            (value for prefix, value in replacements.items() if line.startswith(prefix)), None
        )
        updated.append(replacement or line)
    atomic_write_text(path, "\n".join(updated) + "\n")


def test_model(
    version: str, season: int = 2025, settings: Settings | None = None
) -> dict[str, Any]:
    if season != 2025:
        raise ValueError("V1's untouched test is permanently fixed to season 2025")
    resolved = settings or get_settings()
    initialize_database(resolved)
    artifact_path = resolved.artifacts_dir / "models" / version / "candidate.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Frozen model {version} does not exist")
    if (artifact_path.parent / "invalidation.json").exists():
        raise RuntimeError(f"Frozen model {version} was invalidated and cannot consume 2025")
    promotion_path = artifact_path.parent / "promotion.json"
    if promotion_path.exists():
        raise RuntimeError(
            f"Model {version} has already consumed the untouched {season} test"
        )
    expected_report_paths = (
        resolved.reports_dir / f"model_{version}_test_{season}.json",
        resolved.reports_dir / f"model_{version}_test_{season}_games.csv",
        resolved.reports_dir / f"model_{version}_test_{season}_gates.csv",
        resolved.reports_dir / f"model_{version}_test_{season}.html",
        resolved.manifests_dir / f"model_{version}_test_{season}.json",
    )
    existing_test_paths = [str(path) for path in expected_report_paths if path.exists()]
    if existing_test_paths:
        raise RuntimeError(
            "Untouched-test output exists without a registered promotion decision; "
            f"refusing to overwrite it: {existing_test_paths}"
        )
    candidate: Candidate = joblib.load(artifact_path)
    if candidate.version != version or candidate.development_end_season != DEVELOPMENT_END_SEASON:
        raise RuntimeError("Frozen candidate metadata does not match the requested test")
    development_matrix = assemble_game_matrix(
        resolved, minimum_season=2010, maximum_season=DEVELOPMENT_END_SEASON
    )
    config = FeatureConfig(candidate.selected_half_life, candidate.selected_prior_strength)
    if candidate.feature_hash != _development_hash(development_matrix, config):
        raise RuntimeError("Development feature artifact changed after model freeze; refusing test")
    development_manifest_path = resolved.manifests_dir / f"model_{version}_development.json"
    if development_manifest_path.exists():
        development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
        if development_manifest.get("artifact_hash") != sha256_bytes(artifact_path.read_bytes()):
            raise RuntimeError("Frozen candidate artifact hash no longer matches its manifest")
    if (resolved.runtime_dir / "features" / "team_game_lags.parquet").exists():
        from nfl_bets.validation import validate_all

        validate_all(resolved)

    started_at = iso_utc()
    try:
        with transaction(resolved) as connection:
            connection.execute(
                "INSERT INTO model_test_registry(model_version,test_season,started_at_utc,status) "
                "VALUES (?,?,?,'STARTED')",
                (version, season, started_at),
            )
    except sqlite3.IntegrityError as exc:
        raise RuntimeError(
            f"Model {version} has already consumed the untouched {season} test"
        ) from exc
    try:
        matrix = assemble_game_matrix(resolved, minimum_season=season, maximum_season=season)
        if matrix.height < 100:
            raise ValueError(f"Untouched {season} sample is incomplete")
        input_hash = _test_input_hash(matrix, candidate)
        x = matrix.select(candidate.feature_columns).to_numpy()
        spread_raw = candidate.spread_model.predict(x)
        total_raw = candidate.total_model.predict(x)
        spread_lines = matrix["spread_line"].to_numpy()
        total_lines = matrix["total_line"].to_numpy()
        spread_final = candidate.spread_market_alpha * spread_raw + (
            1 - candidate.spread_market_alpha
        ) * spread_lines
        total_final = candidate.total_market_alpha * total_raw + (
            1 - candidate.total_market_alpha
        ) * total_lines
        spread_probability = candidate.spread_calibrator.predict(
            _probability_from_mapper(candidate.spread_residuals, spread_final, spread_lines)
        )
        total_probability = candidate.total_calibrator.predict(
            _probability_from_mapper(candidate.total_residuals, total_final, total_lines)
        )
        spread_market_probability = matrix["market_spread_probability"].to_numpy()
        total_market_probability = matrix["market_total_probability"].to_numpy()
        for name, values in (
            ("spread candidate", spread_probability),
            ("total candidate", total_probability),
            ("spread market", spread_market_probability),
            ("total market", total_market_probability),
        ):
            if not np.all(np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
                raise ValueError(f"{name} probabilities are not finite and strictly inside (0, 1)")

        spread_actual = matrix["target_margin"].to_numpy()
        total_actual = matrix["target_total"].to_numpy()
        spread_mask = spread_actual != spread_lines
        total_mask = total_actual != total_lines
        if int(spread_mask.sum()) < 100 or int(total_mask.sum()) < 100:
            raise ValueError("Untouched test has too few non-push outcomes")
        spread_y = (spread_actual[spread_mask] > spread_lines[spread_mask]).astype(int)
        total_y = (total_actual[total_mask] > total_lines[total_mask]).astype(int)
        spread_candidate = _probability_metrics(spread_y, spread_probability[spread_mask])
        total_candidate = _probability_metrics(total_y, total_probability[total_mask])
        spread_calibration = _calibration(spread_y, spread_probability[spread_mask])
        total_calibration = _calibration(total_y, total_probability[total_mask])
        spread_market = _probability_metrics(
            spread_y, spread_market_probability[spread_mask]
        )
        total_market = _probability_metrics(total_y, total_market_probability[total_mask])
        gates = _test_gate_results(
            spread_candidate,
            spread_market,
            spread_calibration,
            total_candidate,
            total_market,
            total_calibration,
        )
        status = "PROMOTED" if all(gates.values()) else "PASS_ONLY"
        completed_at = iso_utc()
        report: dict[str, Any] = {
            "model_version": version,
            "test_season": season,
            "status": status,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "rows": matrix.height,
            "input_hash": input_hash,
            "candidate_artifact_hash": sha256_bytes(artifact_path.read_bytes()),
            "development_feature_hash": candidate.feature_hash,
            "spread": {
                "non_push_rows": int(spread_mask.sum()),
                "push_rows": int((~spread_mask).sum()),
                "candidate": spread_candidate,
                "calibration": spread_calibration,
                "reliability": _reliability(
                    spread_y, spread_probability[spread_mask]
                ),
                "raw": _projection_metrics(spread_actual, spread_raw),
                "market": {
                    **spread_market,
                    **_projection_metrics(spread_actual, spread_lines),
                },
                "market_regressed": _projection_metrics(spread_actual, spread_final),
            },
            "total": {
                "non_push_rows": int(total_mask.sum()),
                "push_rows": int((~total_mask).sum()),
                "candidate": total_candidate,
                "calibration": total_calibration,
                "reliability": _reliability(total_y, total_probability[total_mask]),
                "raw": _projection_metrics(total_actual, total_raw),
                "market": {
                    **total_market,
                    **_projection_metrics(total_actual, total_lines),
                },
                "market_regressed": _projection_metrics(total_actual, total_final),
            },
            "promotion_gates": gates,
            "promotion_rule": (
                "Both Brier and log loss must beat no-vig close for both markets with "
                "calibration intercept within +/-0.20 and slope from 0.50 through 1.50."
            ),
            "policy": "The 2025 result is consumed once and cannot tune V1.",
        }
        audit = matrix.select("game_id", "season", "week").with_columns(
            pl.Series("spread_actual_margin", spread_actual),
            pl.Series("spread_line", spread_lines),
            pl.Series("spread_raw_projection", spread_raw),
            pl.Series("spread_final_projection", spread_final),
            pl.Series("spread_model_probability", spread_probability),
            pl.Series("spread_market_probability", spread_market_probability),
            pl.Series("spread_push", ~spread_mask),
            pl.Series("total_actual", total_actual),
            pl.Series("total_line", total_lines),
            pl.Series("total_raw_projection", total_raw),
            pl.Series("total_final_projection", total_final),
            pl.Series("total_model_probability", total_probability),
            pl.Series("total_market_probability", total_market_probability),
            pl.Series("total_push", ~total_mask),
        )
        report_paths = _write_test_reports(version, season, report, audit, resolved)
        promotion = {
            "model_version": version,
            "test_season": season,
            "status": status,
            "tested_at_utc": completed_at,
            "input_hash": input_hash,
            "promotion_gates": gates,
            "reports": report_paths,
        }
        atomic_write_text(
            promotion_path, json.dumps(promotion, sort_keys=True, indent=2)
        )
        _append_history(
            {
                "model_version": version,
                "created_at_utc": completed_at,
                "command": "test",
                "development_end_season": candidate.development_end_season,
                "test_season": season,
                "spec_hash": candidate.spec_hash,
                "feature_hash": candidate.feature_hash,
                "artifact_path": str(artifact_path),
                "status": status,
                "spread_alpha": candidate.spread_market_alpha,
                "total_alpha": candidate.total_market_alpha,
                "spread_brier": spread_candidate["brier"],
                "spread_log_loss": spread_candidate["log_loss"],
                "spread_market_brier": spread_market["brier"],
                "spread_market_log_loss": spread_market["log_loss"],
                "total_brier": total_candidate["brier"],
                "total_log_loss": total_candidate["log_loss"],
                "total_market_brier": total_market["brier"],
                "total_market_log_loss": total_market["log_loss"],
                "notes": (
                    "Untouched test consumed once; PASS-only is the default on any gate failure."
                ),
                "source": "local:untouched-test",
                "retrieved_at_utc": completed_at,
                "source_updated_at_utc": None,
                "schema_version": resolved.schema_version,
            },
            resolved,
        )
        manifest = {
            "model_version": version,
            "test_season": season,
            "status": status,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "input_hash": input_hash,
            "candidate_artifact_hash": sha256_bytes(artifact_path.read_bytes()),
            "report_hashes": {
                name: sha256_bytes(Path(path).read_bytes())
                for name, path in report_paths.items()
            },
            "promotion_gates": gates,
            "schema_version": resolved.schema_version,
        }
        atomic_write_text(
            resolved.manifests_dir / f"model_{version}_test_{season}.json",
            json.dumps(manifest, sort_keys=True, indent=2),
        )
        _update_test_project_state(version, status, resolved)
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE model_test_registry SET completed_at_utc=?,status=? "
                "WHERE model_version=? AND test_season=?",
                (completed_at, status, version, season),
            )
        return report
    except BaseException as exc:
        failed_at = iso_utc()
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE model_test_registry SET completed_at_utc=?,status='FAILED',error_message=? "
                "WHERE model_version=? AND test_season=?",
                (failed_at, f"{type(exc).__name__}: {exc}", version, season),
            )
        raise
