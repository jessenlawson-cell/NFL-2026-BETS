from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import initialize_database, transaction
from nfl_bets.features.build import METRICS
from nfl_bets.model.residuals import KEY_NUMBERS, EmpiricalResidualMapper, ProbabilityCalibrator
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_text, canonical_hash, iso_utc, sha256_bytes

RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
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
    spread_market_alpha: float
    total_market_alpha: float
    spread_residuals: EmpiricalResidualMapper
    total_residuals: EmpiricalResidualMapper
    spread_calibrator: ProbabilityCalibrator
    total_calibrator: ProbabilityCalibrator
    development_end_season: int
    training_rows: int
    feature_hash: str
    spec_hash: str


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
) -> pl.DataFrame:
    resolved = settings or get_settings()
    team = pl.read_csv(resolved.root / "team_metrics.csv", infer_schema_length=100_000)
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


def _fit_predict_oof(
    matrix: pl.DataFrame, target: str, alpha: float, last_season: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    markets: list[np.ndarray] = []
    seasons: list[np.ndarray] = []
    for validation_season in range(2012, last_season + 1):
        train = matrix.filter(pl.col("season") < validation_season)
        valid = matrix.filter(pl.col("season") == validation_season)
        if train.height < 100 or valid.is_empty():
            continue
        x_train = train.select(FEATURE_COLUMNS).to_numpy()
        y_train = train[target].to_numpy()
        x_valid = valid.select(FEATURE_COLUMNS).to_numpy()
        model = _pipeline(alpha).fit(x_train, y_train)
        predictions.append(model.predict(x_valid))
        actuals.append(valid[target].to_numpy())
        market_column = "spread_line" if target == "target_margin" else "total_line"
        markets.append(valid[market_column].to_numpy())
        seasons.append(valid["season"].to_numpy())
    if not predictions:
        raise ValueError("Insufficient chronological folds for training")
    return tuple(np.concatenate(items) for items in (predictions, actuals, markets, seasons))  # type: ignore[return-value]


def _choose_ridge(
    matrix: pl.DataFrame, target: str, last_season: int
) -> tuple[float, tuple[np.ndarray, ...]]:
    choices: list[tuple[float, float, tuple[np.ndarray, ...]]] = []
    for alpha in RIDGE_GRID:
        arrays = _fit_predict_oof(matrix, target, alpha, last_season)
        choices.append((root_mean_squared_error(arrays[1], arrays[0]), alpha, arrays))
    _, best_alpha, best_arrays = min(choices, key=lambda item: (item[0], item[1]))
    return best_alpha, best_arrays


def _market_weight(raw: np.ndarray, actual: np.ndarray, market: np.ndarray) -> float:
    delta = raw - market
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(delta, actual - market) / denominator, 0.0, 1.0))


def _training_probabilities(
    matrix: pl.DataFrame,
    target: str,
    ridge_alpha: float,
    market_alpha: float,
    development_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    outcomes: list[int] = []
    line_column = "spread_line" if target == "target_margin" else "total_line"
    keys = KEY_NUMBERS if target == "target_margin" else ()
    for season in range(2012, development_end + 1):
        train = matrix.filter(pl.col("season") < season)
        valid = matrix.filter(pl.col("season") == season)
        if train.height < 100 or valid.is_empty():
            continue
        model = _pipeline(ridge_alpha).fit(
            train.select(FEATURE_COLUMNS).to_numpy(), train[target].to_numpy()
        )
        raw_train = model.predict(train.select(FEATURE_COLUMNS).to_numpy())
        raw_valid = model.predict(valid.select(FEATURE_COLUMNS).to_numpy())
        train_market = train[line_column].to_numpy()
        valid_market = valid[line_column].to_numpy()
        final_train = market_alpha * raw_train + (1.0 - market_alpha) * train_market
        final_valid = market_alpha * raw_valid + (1.0 - market_alpha) * valid_market
        mapper = EmpiricalResidualMapper(key_numbers=keys).fit(
            train[target].to_numpy(), final_train
        )
        for actual, projection, line in zip(
            valid[target].to_numpy(), final_valid, valid_market, strict=True
        ):
            result = mapper.probabilities(float(projection), float(line))
            if actual == line:
                continue
            probabilities.append(result.win / (result.win + result.loss))
            outcomes.append(int(actual > line))
    return np.asarray(probabilities), np.asarray(outcomes)


def _hash_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.exists() else "MISSING"


def _model_data_hash(settings: Settings) -> str:
    manifest = _hash_file(settings.root / "team_metrics.csv") + _hash_file(
        settings.root / "games.csv"
    )
    return sha256_bytes(manifest.encode("ascii"))


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
            for row in connection.execute("SELECT * FROM model_history ORDER BY created_at_utc")
        ]
    frame = pl.DataFrame(rows).select(columns) if rows else pl.DataFrame({c: [] for c in columns})
    atomic_write_text(settings.root / "model_history.csv", frame.write_csv())
    atomic_write_text(settings.curated_dir / "model_history.csv", frame.write_csv())


def train_model(version: str, settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    artifact_path = resolved.artifacts_dir / "models" / version / "candidate.joblib"
    if artifact_path.exists():
        raise FileExistsError(f"Model version {version} is already frozen")
    matrix = assemble_game_matrix(resolved, minimum_season=2010, maximum_season=2024)
    maximum_season = matrix.select(pl.col("season").max()).item()
    if not isinstance(maximum_season, int) or maximum_season < 2024:
        raise ValueError("Development data must include the 2024 regular season")

    spread_ridge, spread_oof = _choose_ridge(matrix, "target_margin", 2024)
    total_ridge, total_oof = _choose_ridge(matrix, "target_total", 2024)
    spread_market_alpha = _market_weight(spread_oof[0], spread_oof[1], spread_oof[2])
    total_market_alpha = _market_weight(total_oof[0], total_oof[1], total_oof[2])
    spread_probs, spread_outcomes = _training_probabilities(
        matrix, "target_margin", spread_ridge, spread_market_alpha, 2024
    )
    total_probs, total_outcomes = _training_probabilities(
        matrix, "target_total", total_ridge, total_market_alpha, 2024
    )
    spread_calibrator = ProbabilityCalibrator().fit(spread_probs, spread_outcomes)
    total_calibrator = ProbabilityCalibrator().fit(total_probs, total_outcomes)

    x = matrix.select(FEATURE_COLUMNS).to_numpy()
    spread_y = matrix["target_margin"].to_numpy()
    total_y = matrix["target_total"].to_numpy()
    spread_model = _pipeline(spread_ridge).fit(x, spread_y)
    total_model = _pipeline(total_ridge).fit(x, total_y)
    spread_raw = spread_model.predict(x)
    total_raw = total_model.predict(x)
    spread_final = (
        spread_market_alpha * spread_raw
        + (1 - spread_market_alpha) * matrix["spread_line"].to_numpy()
    )
    total_final = (
        total_market_alpha * total_raw + (1 - total_market_alpha) * matrix["total_line"].to_numpy()
    )
    feature_hash = _model_data_hash(resolved)
    spec_hash = _hash_file(resolved.root / "MODEL_SPEC.md")
    candidate = Candidate(
        version=version,
        feature_columns=FEATURE_COLUMNS,
        spread_model=spread_model,
        total_model=total_model,
        spread_market_alpha=spread_market_alpha,
        total_market_alpha=total_market_alpha,
        spread_residuals=EmpiricalResidualMapper(key_numbers=KEY_NUMBERS).fit(
            spread_y, spread_final
        ),
        total_residuals=EmpiricalResidualMapper().fit(total_y, total_final),
        spread_calibrator=spread_calibrator,
        total_calibrator=total_calibrator,
        development_end_season=2024,
        training_rows=matrix.height,
        feature_hash=feature_hash,
        spec_hash=spec_hash,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=False)
    joblib.dump(candidate, artifact_path)
    metadata = {
        "version": version,
        "status": "LOCKED_UNTESTED",
        "created_at_utc": iso_utc(),
        "development_end_season": 2024,
        "training_rows": matrix.height,
        "spread_ridge_alpha": spread_ridge,
        "total_ridge_alpha": total_ridge,
        "spread_market_alpha": spread_market_alpha,
        "total_market_alpha": total_market_alpha,
        "feature_hash": feature_hash,
        "spec_hash": spec_hash,
    }
    atomic_write_text(
        artifact_path.parent / "metadata.json", json.dumps(metadata, sort_keys=True, indent=2)
    )
    created_at = iso_utc()
    _append_history(
        {
            "model_version": version,
            "created_at_utc": created_at,
            "command": "train",
            "development_end_season": 2024,
            "test_season": None,
            "spec_hash": spec_hash,
            "feature_hash": feature_hash,
            "artifact_path": str(artifact_path),
            "status": "LOCKED_UNTESTED",
            "spread_alpha": spread_market_alpha,
            "total_alpha": total_market_alpha,
            "spread_brier": None,
            "spread_log_loss": None,
            "spread_market_brier": None,
            "spread_market_log_loss": None,
            "total_brier": None,
            "total_log_loss": None,
            "total_market_brier": None,
            "total_market_log_loss": None,
            "notes": "Hyperparameters and calibration locked before 2025 test.",
            "source": "local:chronological-training",
            "retrieved_at_utc": created_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    return metadata


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return {
        "brier": float(brier_score_loss(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def _calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs").fit(logits, y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def _calibration_is_acceptable(values: dict[str, float]) -> bool:
    return abs(values["intercept"]) <= 0.20 and 0.50 <= values["slope"] <= 1.50


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
    candidate: Candidate = joblib.load(artifact_path)
    if candidate.feature_hash != _model_data_hash(resolved):
        raise RuntimeError("Feature artifact changed after model freeze; refusing test")
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
    matrix = assemble_game_matrix(resolved, minimum_season=season, maximum_season=season)
    if matrix.height < 100:
        raise ValueError(f"Untouched {season} sample is incomplete")
    x = matrix.select(candidate.feature_columns).to_numpy()
    spread_raw = candidate.spread_model.predict(x)
    total_raw = candidate.total_model.predict(x)
    spread_lines = matrix["spread_line"].to_numpy()
    total_lines = matrix["total_line"].to_numpy()
    spread_final = (
        candidate.spread_market_alpha * spread_raw
        + (1 - candidate.spread_market_alpha) * spread_lines
    )
    total_final = (
        candidate.total_market_alpha * total_raw + (1 - candidate.total_market_alpha) * total_lines
    )

    spread_probability = np.array(
        [
            (lambda value: value.win / (value.win + value.loss))(
                candidate.spread_residuals.probabilities(float(projection), float(line))
            )
            for projection, line in zip(spread_final, spread_lines, strict=True)
        ]
    )
    total_probability = np.array(
        [
            (lambda value: value.win / (value.win + value.loss))(
                candidate.total_residuals.probabilities(float(projection), float(line))
            )
            for projection, line in zip(total_final, total_lines, strict=True)
        ]
    )
    spread_probability = candidate.spread_calibrator.predict(spread_probability)
    total_probability = candidate.total_calibrator.predict(total_probability)
    spread_actual = matrix["target_margin"].to_numpy()
    total_actual = matrix["target_total"].to_numpy()
    spread_mask = spread_actual != spread_lines
    total_mask = total_actual != total_lines
    spread_y = (spread_actual[spread_mask] > spread_lines[spread_mask]).astype(int)
    total_y = (total_actual[total_mask] > total_lines[total_mask]).astype(int)
    spread_candidate = _metrics(spread_y, spread_probability[spread_mask])
    total_candidate = _metrics(total_y, total_probability[total_mask])
    spread_calibration = _calibration(spread_y, spread_probability[spread_mask])
    total_calibration = _calibration(total_y, total_probability[total_mask])
    spread_market = _metrics(spread_y, matrix["market_spread_probability"].to_numpy()[spread_mask])
    total_market = _metrics(total_y, matrix["market_total_probability"].to_numpy()[total_mask])
    promoted = (
        spread_candidate["brier"] < spread_market["brier"]
        and spread_candidate["log_loss"] < spread_market["log_loss"]
        and total_candidate["brier"] < total_market["brier"]
        and total_candidate["log_loss"] < total_market["log_loss"]
        and _calibration_is_acceptable(spread_calibration)
        and _calibration_is_acceptable(total_calibration)
    )
    status = "PROMOTED" if promoted else "PASS_ONLY"
    report = {
        "model_version": version,
        "test_season": season,
        "status": status,
        "rows": matrix.height,
        "spread": {
            "candidate": spread_candidate,
            "market": spread_market,
            "calibration": spread_calibration,
            "raw_rmse": float(root_mean_squared_error(spread_actual, spread_raw)),
            "final_rmse": float(root_mean_squared_error(spread_actual, spread_final)),
            "raw_mae": float(mean_absolute_error(spread_actual, spread_raw)),
            "final_mae": float(mean_absolute_error(spread_actual, spread_final)),
        },
        "total": {
            "candidate": total_candidate,
            "market": total_market,
            "calibration": total_calibration,
            "raw_rmse": float(root_mean_squared_error(total_actual, total_raw)),
            "final_rmse": float(root_mean_squared_error(total_actual, total_final)),
            "raw_mae": float(mean_absolute_error(total_actual, total_raw)),
            "final_mae": float(mean_absolute_error(total_actual, total_final)),
        },
        "promotion_rule": (
            "Both Brier and log loss must beat no-vig close for both markets; "
            "calibration intercept must be within +/-0.20 and slope within [0.50, 1.50]."
        ),
    }
    report_path = resolved.reports_dir / f"model_{version}_test_{season}.json"
    atomic_write_text(report_path, json.dumps(report, sort_keys=True, indent=2))
    atomic_write_text(
        artifact_path.parent / "promotion.json",
        json.dumps(
            {"status": status, "tested_at_utc": iso_utc(), "report": str(report_path)}, indent=2
        ),
    )
    created_at = iso_utc()
    _append_history(
        {
            "model_version": version,
            "created_at_utc": created_at,
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
            "notes": "Untouched test consumed once; PASS-only is the default on any gate failure.",
            "source": "local:untouched-test",
            "retrieved_at_utc": created_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    with transaction(resolved) as connection:
        connection.execute(
            "UPDATE model_test_registry SET completed_at_utc=?,status=? "
            "WHERE model_version=? AND test_season=?",
            (created_at, status, version, season),
        )
    return report
