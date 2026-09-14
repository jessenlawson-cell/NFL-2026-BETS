from __future__ import annotations

import html
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    materialize_feature_configuration,
    validate_lag_boundaries,
)
from nfl_bets.features.v11 import V11_METRICS
from nfl_bets.model.residuals import KEY_NUMBERS, ProbabilityCalibrator, StratifiedResidualMapper
from nfl_bets.model.training import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    RIDGE_GRID,
    FeatureConfig,
    _append_history,
    _calibration,
    _market_fair_probability,
    _probability_from_mapper,
    _probability_metrics,
    _projection_metrics,
    _reliability,
)
from nfl_bets.util import atomic_write_text, iso_utc, sha256_bytes

V11_DEVELOPMENT_START_SEASON = 2014
V11_DEVELOPMENT_END_SEASON = 2025
V11_UNTOUCHED_SEASON = 2026
V11_FEATURE_COLUMNS = (
    "pass_epa_matchup_difference",
    "pass_epa_matchup_sum",
    "rush_epa_matchup_difference",
    "rush_epa_matchup_sum",
    "pass_success_matchup_difference",
    "pass_success_matchup_sum",
    "rush_success_matchup_difference",
    "rush_success_matchup_sum",
    "neutral_rush_epa_matchup_difference",
    "neutral_rush_epa_matchup_sum",
    "neutral_rush_success_matchup_difference",
    "neutral_rush_success_matchup_sum",
    "sack_risk_difference",
    "sack_risk_sum",
    "qb_hit_risk_difference",
    "qb_hit_risk_sum",
    "off_snap_continuity_difference",
    "off_snap_continuity_sum",
    "def_snap_continuity_difference",
    "def_snap_continuity_sum",
    "rest_difference",
)


@dataclass
class V11Candidate:
    version: str
    feature_columns: tuple[str, ...]
    spread_adjustment_model: Pipeline
    total_adjustment_model: Pipeline
    spread_ridge_alpha: float
    total_ridge_alpha: float
    spread_adjustment_weight: float
    total_adjustment_weight: float
    selected_half_life: float
    selected_prior_strength: float
    spread_residuals: StratifiedResidualMapper
    total_residuals: StratifiedResidualMapper
    spread_calibrator: ProbabilityCalibrator
    total_calibrator: ProbabilityCalibrator
    development_end_season: int
    prospective_test_season: int
    prospective_start_utc: str
    training_rows: int
    feature_hash: str
    spec_hash: str


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _matchup_expressions() -> list[pl.Expr]:
    pairs = {
        "pass_epa_matchup": (
            pl.col("home_off_pass_epa") + pl.col("away_def_pass_epa"),
            pl.col("away_off_pass_epa") + pl.col("home_def_pass_epa"),
        ),
        "rush_epa_matchup": (
            pl.col("home_off_rush_epa") + pl.col("away_def_rush_epa"),
            pl.col("away_off_rush_epa") + pl.col("home_def_rush_epa"),
        ),
        "pass_success_matchup": (
            pl.col("home_off_pass_success_rate") + pl.col("away_def_pass_success_rate"),
            pl.col("away_off_pass_success_rate") + pl.col("home_def_pass_success_rate"),
        ),
        "rush_success_matchup": (
            pl.col("home_off_rush_success_rate") + pl.col("away_def_rush_success_rate"),
            pl.col("away_off_rush_success_rate") + pl.col("home_def_rush_success_rate"),
        ),
        "neutral_rush_epa_matchup": (
            pl.col("home_off_neutral_rush_epa") + pl.col("away_def_neutral_rush_epa"),
            pl.col("away_off_neutral_rush_epa") + pl.col("home_def_neutral_rush_epa"),
        ),
        "neutral_rush_success_matchup": (
            pl.col("home_off_neutral_rush_success_rate")
            + pl.col("away_def_neutral_rush_success_rate"),
            pl.col("away_off_neutral_rush_success_rate")
            + pl.col("home_def_neutral_rush_success_rate"),
        ),
        "sack_risk": (
            pl.col("home_off_sack_rate") + pl.col("away_def_sack_rate"),
            pl.col("away_off_sack_rate") + pl.col("home_def_sack_rate"),
        ),
        "qb_hit_risk": (
            pl.col("home_off_qb_hit_rate") + pl.col("away_def_qb_hit_rate"),
            pl.col("away_off_qb_hit_rate") + pl.col("home_def_qb_hit_rate"),
        ),
        "off_snap_continuity": (
            pl.col("home_off_snap_continuity"),
            pl.col("away_off_snap_continuity"),
        ),
        "def_snap_continuity": (
            pl.col("home_def_snap_continuity"),
            pl.col("away_def_snap_continuity"),
        ),
    }
    expressions: list[pl.Expr] = []
    for name, (home, away) in pairs.items():
        expressions.extend(
            [
                (home - away).alias(f"{name}_difference"),
                (home + away).alias(f"{name}_sum"),
            ]
        )
    return expressions


def assemble_v11_matrix(
    team_metrics: pl.DataFrame,
    settings: Settings | None = None,
    maximum_season: int = V11_DEVELOPMENT_END_SEASON,
) -> pl.DataFrame:
    if maximum_season >= V11_UNTOUCHED_SEASON:
        raise RuntimeError("V1.1 development cannot read 2026 outcomes")
    resolved = settings or get_settings()
    games = (
        pl.scan_csv(resolved.root / "games.csv", infer_schema_length=100_000)
        .filter(
            (pl.col("season") >= V11_DEVELOPMENT_START_SEASON)
            & (pl.col("season") <= maximum_season)
            & (pl.col("game_type") == "REG")
        )
        .collect()
        .drop_nulls(
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
    )
    team = team_metrics.filter(
        (pl.col("season") >= V11_DEVELOPMENT_START_SEASON)
        & (pl.col("season") <= maximum_season)
    )
    home = team.filter(pl.col("is_home")).select(
        "game_id", *[pl.col(metric).alias(f"home_{metric}") for metric in V11_METRICS]
    )
    away = team.filter(~pl.col("is_home")).select(
        "game_id", *[pl.col(metric).alias(f"away_{metric}") for metric in V11_METRICS]
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
            *_matchup_expressions(),
        )
        .drop_nulls(
            [
                *V11_FEATURE_COLUMNS,
                "target_margin",
                "target_total",
                "market_spread_probability",
                "market_total_probability",
            ]
        )
        .sort(["season", "week", "kickoff_utc", "game_id"])
    )
    if matrix.is_empty() or cast(int, matrix["season"].max()) != maximum_season:
        raise ValueError(f"V1.1 development matrix must reach {maximum_season}")
    return matrix


def _development_hash(matrix: pl.DataFrame, config: FeatureConfig) -> str:
    columns = [
        "game_id",
        "season",
        *V11_FEATURE_COLUMNS,
        "target_margin",
        "target_total",
        "spread_line",
        "total_line",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    ]
    payload = matrix.select(columns).sort(["season", "game_id"]).write_csv().encode()
    settings = f"{config.half_life:.6f}|{config.prior_strength:.6f}".encode()
    return sha256_bytes(settings + b"\n" + payload)


def _validate_matrix_coverage(matrix: pl.DataFrame, settings: Settings) -> None:
    eligible = (
        pl.scan_csv(settings.root / "games.csv", infer_schema_length=100_000)
        .filter(
            (pl.col("season") >= V11_DEVELOPMENT_START_SEASON)
            & (pl.col("season") <= V11_DEVELOPMENT_END_SEASON)
            & (pl.col("game_type") == "REG")
        )
        .collect()
        .drop_nulls(
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
        .group_by("season")
        .len()
        .rename({"len": "eligible"})
    )
    included = matrix.group_by("season").len().rename({"len": "included"})
    breaches = (
        eligible.join(included, on="season", how="left", validate="1:1")
        .with_columns(pl.col("included").fill_null(0))
        .with_columns(
            ((pl.col("eligible") - pl.col("included")) / pl.col("eligible")).alias(
                "excluded_rate"
            )
        )
        .filter(pl.col("excluded_rate") > 0.05)
    )
    if breaches.height:
        raise ValueError(
            "V1.1 feature exclusions exceed 5% in a development season: "
            f"{breaches.to_dicts()}"
        )


def _oof_adjustments(
    matrix: pl.DataFrame,
    target: str,
    line_column: str,
    probability_column: str,
    ridge_alpha: float,
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for validation_season in range(V11_DEVELOPMENT_START_SEASON + 1, 2026):
        train = matrix.filter(pl.col("season") < validation_season)
        valid = matrix.filter(pl.col("season") == validation_season)
        if train.height < 100 or valid.is_empty():
            continue
        model = _pipeline(ridge_alpha).fit(
            train.select(V11_FEATURE_COLUMNS).to_numpy(),
            (train[target] - train[line_column]).to_numpy(),
        )
        adjustment = model.predict(valid.select(V11_FEATURE_COLUMNS).to_numpy())
        rows.append(
            valid.select(
                "game_id", "season", target, line_column, probability_column
            ).with_columns(
                pl.Series("raw_adjustment", adjustment)
            )
        )
    if not rows:
        raise ValueError("Insufficient chronological V1.1 folds")
    return pl.concat(rows).rename(
        {
            target: "actual",
            line_column: "market_projection",
            probability_column: "market_probability",
        }
    )


def _adjustment_weight(frame: pl.DataFrame) -> float:
    adjustment = frame["raw_adjustment"].to_numpy()
    residual = frame["actual"].to_numpy() - frame["market_projection"].to_numpy()
    denominator = float(np.dot(adjustment, adjustment))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(adjustment, residual) / denominator, 0.0, 1.0))


def _select_configuration(
    caches: dict[str, dict[str, dict[float, pl.DataFrame]]],
    configs: dict[str, FeatureConfig],
    before_season: int | None,
) -> tuple[FeatureConfig, dict[str, float], dict[str, pl.DataFrame]]:
    choices: list[
        tuple[float, float, float, FeatureConfig, dict[str, float], dict[str, pl.DataFrame]]
    ] = []
    for key, config in configs.items():
        target_ridges: dict[str, float] = {}
        target_frames: dict[str, pl.DataFrame] = {}
        normalized: list[float] = []
        for target in ("spread", "total"):
            ridge_choices: list[tuple[float, float, pl.DataFrame]] = []
            for ridge, frame in caches[key][target].items():
                selected = (
                    frame
                    if before_season is None
                    else frame.filter(pl.col("season") < before_season)
                )
                if selected.height < 100:
                    continue
                weight = _adjustment_weight(selected)
                final = (
                    selected["market_projection"] + weight * selected["raw_adjustment"]
                )
                rmse = float(root_mean_squared_error(selected["actual"], final))
                ridge_choices.append((rmse, ridge, selected))
            if not ridge_choices:
                break
            rmse, ridge, frame = min(ridge_choices, key=lambda item: (item[0], item[1]))
            target_ridges[target] = ridge
            target_frames[target] = frame
            scale = max(float(cast(float, frame["actual"].std())), 1e-9)
            normalized.append(rmse / scale)
        if len(normalized) != 2:
            continue
        choices.append(
            (
                float(np.mean(normalized)),
                config.half_life,
                config.prior_strength,
                config,
                target_ridges,
                target_frames,
            )
        )
    if not choices:
        raise ValueError("No V1.1 configuration has sufficient earlier OOF data")
    _, _, _, config, ridges, frames = min(choices, key=lambda item: item[:3])
    return config, ridges, frames


def _bootstrap(rows: pl.DataFrame) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seasons = np.asarray(sorted(rows["season"].unique().to_list()), dtype=int)
    season_values = rows["season"].to_numpy()
    by_season = {int(season): np.flatnonzero(season_values == season) for season in seasons}
    samples: dict[str, list[float]] = {
        "spread_final_minus_market_rmse": [],
        "total_final_minus_market_rmse": [],
    }
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = [
            rng.choice(by_season[int(season)], size=len(by_season[int(season)]), replace=True)
            for season in rng.choice(seasons, size=len(seasons), replace=True)
        ]
        indices = np.concatenate(selected)
        for prefix, actual_column, final_column, line_column in (
            ("spread", "target_margin", "spread_final", "spread_line"),
            ("total", "target_total", "total_final", "total_line"),
        ):
            difference = root_mean_squared_error(
                rows[actual_column].to_numpy()[indices], rows[final_column].to_numpy()[indices]
            ) - root_mean_squared_error(
                rows[actual_column].to_numpy()[indices], rows[line_column].to_numpy()[indices]
            )
            samples[f"{prefix}_final_minus_market_rmse"].append(float(difference))
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def _write_reports(
    version: str,
    report: dict[str, Any],
    folds: list[dict[str, Any]],
    coefficients: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, str]:
    prefix = settings.reports_dir / f"model_{version}_development"
    paths = {
        "json": Path(f"{prefix}.json"),
        "html": Path(f"{prefix}.html"),
        "folds_csv": settings.reports_dir / f"model_{version}_development_folds.csv",
        "coefficients_csv": settings.reports_dir / f"model_{version}_coefficients.csv",
    }
    atomic_write_text(paths["json"], json.dumps(report, sort_keys=True, indent=2))
    atomic_write_text(paths["folds_csv"], pl.DataFrame(folds).write_csv())
    atomic_write_text(paths["coefficients_csv"], pl.DataFrame(coefficients).write_csv())
    gate = html.escape(json.dumps(report["development_metrics"], indent=2))
    config = html.escape(json.dumps(report["selected_configuration"], indent=2))
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NFL model {html.escape(version)} development</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#222}}
h1,h2{{color:#17365d}}pre{{background:#f4f7fa;padding:16px;overflow:auto}}</style></head>
<body><h1>NFL Model {html.escape(version)} — V1.1 Development</h1>
<p>Status: <strong>LOCKED_UNTESTED_2026</strong>. Only games after the recorded prospective cutoff
may enter the new untouched evaluation.</p><h2>Selected configuration</h2><pre>{config}</pre>
<h2>Development metrics</h2><pre>{gate}</pre></body></html>"""
    atomic_write_text(paths["html"], document)
    return {name: str(path) for name, path in paths.items()}


def invalidate_v11_candidate(
    version: str,
    reason: str,
    status: str = "INVALID_SPEC",
    settings: Settings | None = None,
) -> dict[str, Any]:
    if status not in {"INVALID_SPEC", "INVALID_METHOD", "INVALID_DATA_QUALITY"}:
        raise ValueError("Unsupported V1.1 invalidation status")
    resolved = settings or get_settings()
    artifact_dir = resolved.artifacts_dir / "models" / version
    metadata_path = artifact_dir / "metadata.json"
    invalidation_path = artifact_dir / "invalidation.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Frozen V1.1 model {version} does not exist")
    if invalidation_path.exists():
        return cast(
            dict[str, Any], json.loads(invalidation_path.read_text(encoding="utf-8"))
        )
    metadata = cast(dict[str, Any], json.loads(metadata_path.read_text(encoding="utf-8")))
    invalidated_at = iso_utc()
    invalidation = {
        "model_version": version,
        "status": status,
        "invalidated_at_utc": invalidated_at,
        "reason": reason,
        "prospective_test_consumed": False,
    }
    atomic_write_text(
        invalidation_path, json.dumps(invalidation, sort_keys=True, indent=2)
    )
    selected = cast(dict[str, Any], metadata.get("selected_configuration", {}))
    _append_history(
        {
            "model_version": version,
            "created_at_utc": invalidated_at,
            "command": "validate-v11",
            "development_end_season": metadata.get("development_end_season"),
            "test_season": None,
            "spec_hash": metadata.get("spec_hash"),
            "feature_hash": metadata.get("feature_hash"),
            "artifact_path": str(artifact_dir / "candidate.joblib"),
            "status": status,
            "spread_alpha": selected.get("spread_adjustment_weight"),
            "total_alpha": selected.get("total_adjustment_weight"),
            "spread_brier": None,
            "spread_log_loss": None,
            "spread_market_brier": None,
            "spread_market_log_loss": None,
            "total_brier": None,
            "total_log_loss": None,
            "total_market_brier": None,
            "total_market_log_loss": None,
            "notes": reason,
            "source": "local:v11-spec-validation",
            "retrieved_at_utc": invalidated_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    return invalidation


def _update_project_state(version: str, prospective_start: str, settings: Settings) -> None:
    path = settings.root / "PROJECT_STATE.md"
    if not path.exists():
        return
    replacements = {
        "- **Last Model Version:**": (
            f"- **Last Model Version:** {version} (LOCKED_UNTESTED_2026)"
        ),
        "- **System Status:**": (
            "- **System Status:** V1 remains PASS-only; V1.1 feature expansion developed through "
            f"2025 and frozen for prospective 2026 evaluation after {prospective_start}"
        ),
        "- **Betting Status:**": (
            "- **Betting Status:** PASS-only; V1.1 has no prospective promotion evidence"
        ),
    }
    updated = []
    for line in path.read_text(encoding="utf-8").splitlines():
        replacement = next(
            (value for prefix, value in replacements.items() if line.startswith(prefix)), None
        )
        updated.append(replacement or line)
    atomic_write_text(path, "\n".join(updated) + "\n")


def train_v11_model(
    version: str = "1.1.0", settings: Settings | None = None
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    if not version.startswith("1.1."):
        raise ValueError("V1.1 candidates must use a 1.1.x version")
    artifact_path = resolved.artifacts_dir / "models" / version / "candidate.joblib"
    if artifact_path.parent.exists():
        raise FileExistsError(f"Model version {version} is already reserved")
    lag_path = resolved.runtime_dir / "features" / "v11_team_game_lags.parquet"
    manifest_path = resolved.manifests_dir / "v11_feature_inputs.latest.json"
    if not lag_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Build V1.1 feature inputs before training")
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("injury_model_status") != (
        "DISABLED_INSUFFICIENT_HISTORICAL_COVERAGE"
    ):
        raise RuntimeError("Unexpected V1.1 injury feature policy")
    all_lags = pl.read_parquet(lag_path)
    if feature_manifest.get("content_hash") != sha256_bytes(lag_path.read_bytes()):
        raise RuntimeError("V1.1 feature store hash does not match its manifest")
    validate_lag_boundaries(all_lags)
    development_lags = all_lags.filter(pl.col("season") <= V11_DEVELOPMENT_END_SEASON)
    if development_lags.filter(pl.col("season") >= V11_UNTOUCHED_SEASON).height:
        raise RuntimeError("2026 feature rows entered V1.1 development")

    configs: dict[str, FeatureConfig] = {}
    matrices: dict[str, pl.DataFrame] = {}
    caches: dict[str, dict[str, dict[float, pl.DataFrame]]] = {}
    for half_life in HALF_LIFE_GRID:
        for prior_strength in PRIOR_STRENGTH_GRID:
            config = FeatureConfig(half_life, prior_strength)
            configs[config.key] = config
            materialized = materialize_feature_configuration(
                development_lags,
                half_life=half_life,
                prior_strength=prior_strength,
                metrics=V11_METRICS,
            )
            matrix = assemble_v11_matrix(materialized, resolved)
            _validate_matrix_coverage(matrix, resolved)
            matrices[config.key] = matrix
            caches[config.key] = {"spread": {}, "total": {}}
            for ridge in RIDGE_GRID:
                caches[config.key]["spread"][ridge] = _oof_adjustments(
                    matrix,
                    "target_margin",
                    "spread_line",
                    "market_spread_probability",
                    ridge,
                )
                caches[config.key]["total"][ridge] = _oof_adjustments(
                    matrix,
                    "target_total",
                    "total_line",
                    "market_total_probability",
                    ridge,
                )

    nested_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for season in range(2017, V11_DEVELOPMENT_END_SEASON + 1):
        try:
            config, ridges, inner = _select_configuration(caches, configs, season)
        except ValueError:
            continue
        matrix = matrices[config.key]
        train = matrix.filter(pl.col("season") < season)
        valid = matrix.filter(pl.col("season") == season)
        if train.height < 100 or valid.is_empty():
            continue
        fold: dict[str, Any] = {
            "season": season,
            "rows": valid.height,
            "half_life": config.half_life,
            "prior_strength": config.prior_strength,
        }
        predictions: dict[str, np.ndarray] = {}
        for prefix, target, line_column in (
            ("spread", "target_margin", "spread_line"),
            ("total", "target_total", "total_line"),
        ):
            model = _pipeline(ridges[prefix]).fit(
                train.select(V11_FEATURE_COLUMNS).to_numpy(),
                (train[target] - train[line_column]).to_numpy(),
            )
            adjustment = model.predict(valid.select(V11_FEATURE_COLUMNS).to_numpy())
            weight = _adjustment_weight(inner[prefix])
            final_projection = valid[line_column].to_numpy() + weight * adjustment
            predictions[f"{prefix}_adjustment"] = adjustment
            predictions[f"{prefix}_final"] = final_projection
            fold[f"{prefix}_ridge_alpha"] = ridges[prefix]
            fold[f"{prefix}_adjustment_weight"] = weight
            fold[f"{prefix}_market_rmse"] = float(
                root_mean_squared_error(valid[target], valid[line_column])
            )
            fold[f"{prefix}_final_rmse"] = float(
                root_mean_squared_error(valid[target], final_projection)
            )
        folds.append(fold)
        for index, row in enumerate(valid.iter_rows(named=True)):
            nested_rows.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "target_margin": row["target_margin"],
                    "target_total": row["target_total"],
                    "spread_line": row["spread_line"],
                    "total_line": row["total_line"],
                    "spread_adjustment": predictions["spread_adjustment"][index],
                    "total_adjustment": predictions["total_adjustment"][index],
                    "spread_final": predictions["spread_final"][index],
                    "total_final": predictions["total_final"][index],
                }
            )
    if not nested_rows or not folds:
        raise ValueError("V1.1 nested development produced no folds")
    nested = pl.DataFrame(nested_rows).sort(["season", "game_id"])
    selected_config, selected_ridges, selected_oof = _select_configuration(
        caches, configs, None
    )
    selected_matrix = matrices[selected_config.key]
    from nfl_bets.validation import validate_all

    validate_all(resolved)
    final_components: dict[str, dict[str, Any]] = {}
    for prefix in ("spread", "total"):
        oof = selected_oof[prefix]
        weight = _adjustment_weight(oof)
        projection = oof["market_projection"].to_numpy() + weight * oof[
            "raw_adjustment"
        ].to_numpy()
        mapper = StratifiedResidualMapper(
            key_numbers=KEY_NUMBERS if prefix == "spread" else (),
            absolute_strata=prefix == "spread",
        ).fit(oof["actual"].to_numpy(), projection, oof["market_projection"].to_numpy())
        probability = _probability_from_mapper(
            mapper, projection, oof["market_projection"].to_numpy()
        )
        non_push = oof["actual"].to_numpy() != oof["market_projection"].to_numpy()
        outcomes = (
            oof["actual"].to_numpy()[non_push]
            > oof["market_projection"].to_numpy()[non_push]
        ).astype(int)
        calibrator = ProbabilityCalibrator().fit(probability[non_push], outcomes)
        final_components[prefix] = {
            "weight": weight,
            "mapper": mapper,
            "calibrator": calibrator,
            "probability": calibrator.predict(probability[non_push]),
            "outcomes": outcomes,
            "non_push": non_push,
            "oof": oof,
        }

    x = selected_matrix.select(V11_FEATURE_COLUMNS).to_numpy()
    spread_model = _pipeline(selected_ridges["spread"]).fit(
        x, (selected_matrix["target_margin"] - selected_matrix["spread_line"]).to_numpy()
    )
    total_model = _pipeline(selected_ridges["total"]).fit(
        x, (selected_matrix["target_total"] - selected_matrix["total_line"]).to_numpy()
    )
    feature_hash = _development_hash(selected_matrix, selected_config)
    spec_path = resolved.root / "MODEL_SPEC_V1_1.md"
    if not spec_path.exists():
        raise FileNotFoundError("MODEL_SPEC_V1_1.md is required before freezing V1.1")
    spec_hash = sha256_bytes(spec_path.read_bytes())
    prospective_start = iso_utc()
    candidate = V11Candidate(
        version=version,
        feature_columns=V11_FEATURE_COLUMNS,
        spread_adjustment_model=spread_model,
        total_adjustment_model=total_model,
        spread_ridge_alpha=selected_ridges["spread"],
        total_ridge_alpha=selected_ridges["total"],
        spread_adjustment_weight=final_components["spread"]["weight"],
        total_adjustment_weight=final_components["total"]["weight"],
        selected_half_life=selected_config.half_life,
        selected_prior_strength=selected_config.prior_strength,
        spread_residuals=final_components["spread"]["mapper"],
        total_residuals=final_components["total"]["mapper"],
        spread_calibrator=final_components["spread"]["calibrator"],
        total_calibrator=final_components["total"]["calibrator"],
        development_end_season=V11_DEVELOPMENT_END_SEASON,
        prospective_test_season=V11_UNTOUCHED_SEASON,
        prospective_start_utc=prospective_start,
        training_rows=selected_matrix.height,
        feature_hash=feature_hash,
        spec_hash=spec_hash,
    )
    metrics: dict[str, Any] = {}
    for prefix, actual_column, final_column, line_column in (
        ("spread", "target_margin", "spread_final", "spread_line"),
        ("total", "target_total", "total_final", "total_line"),
    ):
        actual = nested[actual_column].to_numpy()
        market = nested[line_column].to_numpy()
        projection = nested[final_column].to_numpy()
        oof = cast(pl.DataFrame, final_components[prefix]["oof"])
        non_push = cast(np.ndarray, final_components[prefix]["non_push"])
        outcomes = cast(np.ndarray, final_components[prefix]["outcomes"])
        probabilities = cast(np.ndarray, final_components[prefix]["probability"])
        market_probability = oof["market_probability"].to_numpy()[non_push]
        metrics[prefix] = {
            "nested_market": _projection_metrics(actual, market),
            "nested_market_plus_adjustment": _projection_metrics(actual, projection),
            "selected_oof_probability": _probability_metrics(outcomes, probabilities),
            "selected_oof_market_probability": _probability_metrics(
                outcomes, market_probability
            ),
            "selected_oof_calibration": _calibration(outcomes, probabilities),
            "selected_oof_reliability": _reliability(outcomes, probabilities),
            "selected_oof_rows": oof.height,
            "selected_oof_non_push_rows": int(non_push.sum()),
        }
    coefficients = [
        {
            "feature": feature,
            "spread_standardized_adjustment": float(
                spread_model.named_steps["ridge"].coef_[index]
            ),
            "total_standardized_adjustment": float(
                total_model.named_steps["ridge"].coef_[index]
            ),
        }
        for index, feature in enumerate(V11_FEATURE_COLUMNS)
    ]
    report: dict[str, Any] = {
        "model_version": version,
        "status": "LOCKED_UNTESTED_2026",
        "development_seasons": [V11_DEVELOPMENT_START_SEASON, V11_DEVELOPMENT_END_SEASON],
        "prospective_test_season": V11_UNTOUCHED_SEASON,
        "prospective_start_utc": prospective_start,
        "prospective_rule": (
            "Only games with kickoff after prospective_start_utc and a prediction timestamp "
            "before kickoff may enter evaluation."
        ),
        "training_rows": selected_matrix.height,
        "selected_configuration": {
            "half_life": selected_config.half_life,
            "prior_strength": selected_config.prior_strength,
            "spread_ridge_alpha": selected_ridges["spread"],
            "total_ridge_alpha": selected_ridges["total"],
            "spread_adjustment_weight": candidate.spread_adjustment_weight,
            "total_adjustment_weight": candidate.total_adjustment_weight,
        },
        "feature_columns": list(V11_FEATURE_COLUMNS),
        "injury_model_status": feature_manifest["injury_model_status"],
        "development_metrics": metrics,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "intervals": _bootstrap(nested),
        },
        "folds": folds,
        "coefficients": coefficients,
        "feature_hash": feature_hash,
        "spec_hash": spec_hash,
    }
    report_paths = _write_reports(version, report, folds, coefficients, resolved)
    staging = resolved.artifacts_dir / "models" / f".staging-{version}-{uuid.uuid4()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        joblib.dump(candidate, staging / "candidate.joblib")
        metadata = {
            "version": version,
            "status": "LOCKED_UNTESTED_2026",
            "created_at_utc": prospective_start,
            "development_end_season": V11_DEVELOPMENT_END_SEASON,
            "prospective_test_season": V11_UNTOUCHED_SEASON,
            "prospective_start_utc": prospective_start,
            "training_rows": selected_matrix.height,
            "selected_configuration": report["selected_configuration"],
            "feature_hash": feature_hash,
            "spec_hash": spec_hash,
            "reports": report_paths,
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
            "created_at_utc": prospective_start,
            "command": "train-v11",
            "development_end_season": V11_DEVELOPMENT_END_SEASON,
            "test_season": None,
            "spec_hash": spec_hash,
            "feature_hash": feature_hash,
            "artifact_path": str(artifact_path),
            "status": "LOCKED_UNTESTED_2026",
            "spread_alpha": candidate.spread_adjustment_weight,
            "total_alpha": candidate.total_adjustment_weight,
            "spread_brier": None,
            "spread_log_loss": None,
            "spread_market_brier": None,
            "spread_market_log_loss": None,
            "total_brier": None,
            "total_log_loss": None,
            "total_market_brier": None,
            "total_market_log_loss": None,
            "notes": (
                "V1.1 secondary adjustment developed through 2025; only post-freeze 2026 games "
                "are prospective evaluation data. Injury diagnostics remain model-disabled."
            ),
            "source": "local:v11-chronological-training",
            "retrieved_at_utc": prospective_start,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        },
        resolved,
    )
    manifest = {
        "model_version": version,
        "status": "LOCKED_UNTESTED_2026",
        "artifact": str(artifact_path.relative_to(resolved.root)).replace("\\", "/"),
        "artifact_hash": sha256_bytes(artifact_path.read_bytes()),
        "metadata_hash": sha256_bytes((artifact_path.parent / "metadata.json").read_bytes()),
        "feature_input_hash": feature_manifest["content_hash"],
        "development_feature_hash": feature_hash,
        "spec_hash": spec_hash,
        "development_end_season": V11_DEVELOPMENT_END_SEASON,
        "prospective_test_season": V11_UNTOUCHED_SEASON,
        "prospective_start_utc": prospective_start,
        "schema_version": resolved.schema_version,
    }
    atomic_write_text(
        resolved.manifests_dir / f"model_{version}_development.json",
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    _update_project_state(version, prospective_start, resolved)
    return metadata
