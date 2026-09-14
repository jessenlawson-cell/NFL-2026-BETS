from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database, table_columns
from nfl_bets.features.build import METRICS, validate_lag_boundaries
from nfl_bets.features.v11 import V11_METRICS
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_text, canonical_hash, sha256_bytes


class DataQualityError(RuntimeError):
    pass


INTEGER_COLUMNS = {
    "season",
    "week",
    "is_home",
    "games_available",
    "american_price",
    "price",
    "closing_price",
    "pinnacle_orientation_price",
    "pinnacle_other_price",
    "closing_orientation_price",
    "closing_other_price",
    "retail_books_count",
    "eligible_non_push",
}
FLOAT_COLUMNS = {
    "away_score",
    "home_score",
    "result",
    "total",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "temp",
    "wind",
    "point",
    "canonical_line",
    "decimal_price",
    "implied_probability",
    "vig_free_probability",
    "overround",
    "snap_share_impact",
    "replacement_quality",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "special_teams_snaps",
    "special_teams_pct",
    "metric_value",
    "spread_alpha",
    "total_alpha",
    "spread_brier",
    "spread_log_loss",
    "spread_market_brier",
    "spread_market_log_loss",
    "total_brier",
    "total_log_loss",
    "total_market_brier",
    "total_market_log_loss",
    "line",
    "stake",
    "bankroll",
    "model_probability",
    "market_fair_probability",
    "expected_roi",
    "closing_line",
    "vig_free_closing_probability",
    "probability_clv",
    "profit_loss",
    "half_life",
    "pinnacle_line",
    "pinnacle_orientation_no_vig_probability",
    "pinnacle_overround",
    "consensus_line",
    "consensus_probability",
    "raw_adjustment",
    "adjustment_weight",
    "final_projection",
    "raw_non_push_win_probability",
    "calibrated_non_push_win_probability",
    "model_win_probability",
    "model_push_probability",
    "model_loss_probability",
    "closing_market_probability",
    "model_non_push_win_probability",
    "actual_value",
    "model_brier",
    "market_brier",
    "model_log_loss",
    "market_log_loss",
    "projection_error",
    "market_projection_error",
    "line_clv",
    *METRICS,
}
TIMESTAMP_COLUMNS = {
    "kickoff_utc",
    "feature_as_of_utc",
    "commence_time_utc",
    "last_update_utc",
    "created_at_utc",
    "timestamp",
    "data_timestamp",
    "retrieved_at_utc",
    "source_updated_at_utc",
    "prediction_created_at_utc",
    "snapshot_retrieved_at_utc",
    "pinnacle_updated_at_utc",
    "settled_at_utc",
}


def _validate_casts(name: str, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    for column in frame.columns:
        try:
            if column in INTEGER_COLUMNS:
                frame.select(pl.col(column).drop_nulls().cast(pl.Int64, strict=True))
            elif column in FLOAT_COLUMNS:
                frame.select(pl.col(column).drop_nulls().cast(pl.Float64, strict=True))
        except Exception as exc:
            raise DataQualityError(f"{name}.csv has invalid numeric dtype in {column}") from exc
    for column in TIMESTAMP_COLUMNS.intersection(frame.columns):
        invalid = 0
        for value in frame[column].drop_nulls().to_list():
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                invalid += 1
                continue
            if parsed.tzinfo is None:
                invalid += 1
        if invalid:
            raise DataQualityError(
                f"{name}.csv has {invalid} invalid or timezone-naive values in {column}"
            )


def _validate_hashes(name: str, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    invalid = 0
    for row in frame.iter_rows(named=True):
        if row["content_hash"] != canonical_hash(row):
            invalid += 1
    if invalid:
        raise DataQualityError(f"{name}.csv has {invalid} unstable or invalid content hashes")


def _validate_model_coverage(games: pl.DataFrame, metrics: pl.DataFrame) -> dict[str, Any]:
    if games.is_empty() or metrics.is_empty():
        return {"status": "NOT_READY", "excluded_by_season": {}}
    completed = games.filter(
        (pl.col("game_type") == "REG")
        & pl.col("season").is_between(2010, 2025)
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    metric_complete = (
        metrics.with_columns(
            pl.all_horizontal([pl.col(metric).is_not_null() for metric in METRICS]).alias(
                "metrics_complete"
            )
        )
        .group_by("game_id")
        .agg(
            pl.len().alias("team_rows"),
            pl.col("metrics_complete").all().alias("metrics_complete"),
        )
    )
    required_market = [
        "home_rest",
        "away_rest",
        "spread_line",
        "total_line",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    ]
    assessed = completed.join(
        metric_complete, on="game_id", how="left", validate="1:1"
    ).with_columns(
        (
            pl.all_horizontal([pl.col(column).is_not_null() for column in required_market])
            & (pl.col("team_rows").fill_null(0) == 2)
            & pl.col("metrics_complete").fill_null(False)
        ).alias("eligible")
    )
    rows: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []
    breaches: list[str] = []
    for row in assessed.filter(~pl.col("eligible")).iter_rows(named=True):
        reasons: list[str] = []
        missing_market = [column for column in required_market if row.get(column) is None]
        if missing_market:
            reasons.append("missing:" + ",".join(missing_market))
        if (row.get("team_rows") or 0) != 2:
            reasons.append("team_row_count")
        if not row.get("metrics_complete"):
            reasons.append("missing_team_metrics")
        exclusions.append(
            {
                "season": row["season"],
                "week": row["week"],
                "game_id": row["game_id"],
                "reasons": ";".join(reasons),
            }
        )
    for season, frame in assessed.partition_by("season", as_dict=True).items():
        season_value = int(season[0] if isinstance(season, tuple) else season)
        total = frame.height
        eligible = int(frame["eligible"].sum())
        excluded = total - eligible
        rate = excluded / total if total else 0.0
        rows[str(season_value)] = {
            "completed_games": total,
            "eligible_games": eligible,
            "excluded_games": excluded,
            "excluded_rate": rate,
        }
        if rate > 0.05:
            breaches.append(
                f"Season {season_value} excludes {rate:.1%} of completed regular-season games"
            )
    return {
        "status": "VALID" if not breaches else "INVALID",
        "excluded_by_season": rows,
        "excluded_games": exclusions,
        "breaches": breaches,
    }


def _validate_v11(settings: Settings) -> dict[str, Any]:
    manifest_path = settings.manifests_dir / "v11_feature_inputs.latest.json"
    if not manifest_path.exists():
        return {"status": "NOT_BUILT"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lag_path = settings.root / str(manifest.get("artifact", ""))
    if not lag_path.exists():
        raise DataQualityError("V1.1 feature manifest points to a missing artifact")
    if manifest.get("content_hash") != sha256_bytes(lag_path.read_bytes()):
        raise DataQualityError("V1.1 feature artifact hash does not match its manifest")
    lags = pl.read_parquet(lag_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        "team_id",
        *[f"lag{offset}_{metric}" for offset in range(1, 5) for metric in V11_METRICS],
    }
    missing = required - set(lags.columns)
    if missing:
        raise DataQualityError(f"V1.1 feature store is missing columns: {sorted(missing)}")
    try:
        validate_lag_boundaries(lags)
    except ValueError as exc:
        raise DataQualityError(str(exc)) from exc
    if lags.select("game_id", "team_id").is_duplicated().any():
        raise DataQualityError("V1.1 feature store contains duplicate team-game keys")
    if lags.group_by("game_id").len().filter(pl.col("len") != 2).height:
        raise DataQualityError("V1.1 feature store must contain exactly two teams per game")
    if manifest.get("injury_model_status") != "DISABLED_INSUFFICIENT_HISTORICAL_COVERAGE":
        raise DataQualityError("V1.1 injury feature status is not safely disabled")

    candidates: dict[str, Any] = {}
    spec_path = settings.root / "MODEL_SPEC_V1_1.md"
    current_spec_hash = sha256_bytes(spec_path.read_bytes()) if spec_path.exists() else None
    for model_manifest_path in sorted(settings.manifests_dir.glob("model_1.1.*_development.json")):
        model = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        version = str(model["model_version"])
        artifact_path = settings.root / str(model["artifact"])
        metadata_path = artifact_path.parent / "metadata.json"
        if not artifact_path.exists() or not metadata_path.exists():
            raise DataQualityError(f"V1.1 candidate {version} is missing frozen files")
        if model.get("artifact_hash") != sha256_bytes(artifact_path.read_bytes()):
            raise DataQualityError(f"V1.1 candidate {version} artifact hash mismatch")
        if model.get("metadata_hash") != sha256_bytes(metadata_path.read_bytes()):
            raise DataQualityError(f"V1.1 candidate {version} metadata hash mismatch")
        invalidation_path = artifact_path.parent / "invalidation.json"
        if invalidation_path.exists():
            invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
            candidates[version] = {"status": invalidation.get("status"), "valid": False}
            continue
        if model.get("spec_hash") != current_spec_hash:
            raise DataQualityError(f"V1.1 candidate {version} specification hash mismatch")
        if model.get("development_end_season") != 2025:
            raise DataQualityError(f"V1.1 candidate {version} has an invalid development boundary")
        if model.get("prospective_test_season") != 2026:
            raise DataQualityError(f"V1.1 candidate {version} has an invalid test season")
        cutoff = datetime.fromisoformat(
            str(model.get("prospective_start_utc", "")).replace("Z", "+00:00")
        )
        if cutoff.tzinfo is None:
            raise DataQualityError(f"V1.1 candidate {version} cutoff is timezone-naive")
        candidates[version] = {"status": model.get("status"), "valid": True}
    return {
        "status": "VALID",
        "rows": lags.height,
        "seasons": manifest.get("seasons"),
        "injury_model_status": manifest.get("injury_model_status"),
        "candidates": candidates,
    }


def _validate_prospective(
    settings: Settings,
    predictions: pl.DataFrame,
    evaluations: pl.DataFrame,
) -> dict[str, Any]:
    policy_path = settings.manifests_dir / "prospective_policy_1.1.2.json"
    model_manifest_path = settings.manifests_dir / "model_1.1.2_development.json"
    if not policy_path.exists():
        if model_manifest_path.exists():
            raise DataQualityError("Frozen model 1.1.2 is missing its prospective policy")
        return {"status": "NOT_CONFIGURED", "predictions": 0, "evaluations": 0}
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != settings.schema_version:
        raise DataQualityError("Prospective policy schema version mismatch")
    if policy.get("allowed_decisions") != ["PASS"]:
        raise DataQualityError("Prospective policy permits a non-PASS decision")
    if not model_manifest_path.exists():
        raise DataQualityError("Prospective policy has no matching frozen model manifest")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    artifact_path = settings.root / str(model_manifest["artifact"])
    metadata_path = artifact_path.parent / "metadata.json"
    if not artifact_path.exists() or not metadata_path.exists():
        raise DataQualityError("Prospective candidate files are missing")
    checks = {
        "artifact_hash": sha256_bytes(artifact_path.read_bytes()),
        "metadata_hash": sha256_bytes(metadata_path.read_bytes()),
        "spec_hash": sha256_bytes((settings.root / "MODEL_SPEC_V1_1.md").read_bytes()),
        "development_feature_hash": model_manifest.get("development_feature_hash"),
        "prospective_start_utc": model_manifest.get("prospective_start_utc"),
    }
    for key, actual in checks.items():
        if policy.get(key) != actual:
            raise DataQualityError(f"Prospective policy mismatch for {key}")
    cutoff = datetime.fromisoformat(str(policy["prospective_start_utc"]).replace("Z", "+00:00"))
    if predictions.height:
        if predictions.filter(pl.col("decision") != "PASS").height:
            raise DataQualityError("Prospective ledger contains a non-PASS decision")
        probability_rows = predictions.drop_nulls(
            [
                "model_win_probability",
                "model_push_probability",
                "model_loss_probability",
            ]
        )
        if probability_rows.filter(
            (
                pl.col("model_win_probability")
                + pl.col("model_push_probability")
                + pl.col("model_loss_probability")
                - 1.0
            ).abs()
            > 1e-9
        ).height:
            raise DataQualityError("Prospective model probabilities do not sum to one")
        temporal = predictions.with_columns(
            pl.col("kickoff_utc").str.to_datetime(time_zone="UTC").alias("kickoff_dt"),
            pl.col("prediction_created_at_utc")
            .str.to_datetime(time_zone="UTC")
            .alias("prediction_dt"),
            pl.col("snapshot_retrieved_at_utc")
            .str.to_datetime(time_zone="UTC")
            .alias("snapshot_dt"),
            pl.col("feature_as_of_utc").str.to_datetime(time_zone="UTC").alias("feature_dt"),
            pl.col("pinnacle_updated_at_utc").str.to_datetime(time_zone="UTC").alias("pinnacle_dt"),
        )
        if temporal.filter(
            (pl.col("prediction_dt") >= pl.col("kickoff_dt"))
            | (pl.col("snapshot_dt") >= pl.col("kickoff_dt"))
            | (pl.col("prediction_dt") < pl.col("snapshot_dt"))
        ).height:
            raise DataQualityError("Prospective prediction violates snapshot/kickoff ordering")
        if temporal.filter(
            pl.col("feature_dt").is_not_null() & (pl.col("feature_dt") >= pl.col("snapshot_dt"))
        ).height:
            raise DataQualityError("Prospective features do not predate their odds snapshot")
        eligible = temporal.filter(pl.col("eligibility_status") == "ELIGIBLE_CLOSE_PROXY")
        if eligible.height:
            required = [
                "pinnacle_line",
                "pinnacle_orientation_no_vig_probability",
                "final_projection",
                "calibrated_non_push_win_probability",
                "pinnacle_dt",
            ]
            if any(eligible[column].null_count() for column in required):
                raise DataQualityError("Eligible prediction is missing scoring fields")
            minutes = (pl.col("kickoff_dt") - pl.col("snapshot_dt")).dt.total_minutes()
            age = (pl.col("snapshot_dt") - pl.col("pinnacle_dt")).dt.total_minutes()
            if eligible.filter(
                (minutes < 5)
                | (minutes > 90)
                | (age < 0)
                | (age > int(policy["quote_max_age_minutes"]))
                | (pl.col("kickoff_dt") <= pl.lit(cutoff))
            ).height:
                raise DataQualityError("Eligible prediction violates the frozen close contract")
    if evaluations.height:
        if (
            evaluations.filter(pl.col("eligible_non_push") == 1)
            .filter(~pl.col("result").is_in(["WIN", "LOSS"]))
            .height
        ):
            raise DataQualityError("Eligible prospective score is not a win or loss")
        scored = evaluations.filter(pl.col("eligible_non_push") == 1)
        score_columns = [
            "model_brier",
            "market_brier",
            "model_log_loss",
            "market_log_loss",
        ]
        if scored.height and any(scored[column].null_count() for column in score_columns):
            raise DataQualityError("Eligible evaluation is missing probability scores")
        pushes = evaluations.filter(pl.col("result") == "PUSH")
        if pushes.height and (
            int(pushes["eligible_non_push"].sum())
            or any(pushes[column].drop_nulls().len() for column in score_columns)
        ):
            raise DataQualityError("Pushes entered prospective probability scoring")
        referenced = evaluations.filter(pl.col("prediction_id").is_not_null())
        if referenced.height:
            unmatched = referenced.select("prediction_id").join(
                predictions.select("prediction_id"), on="prediction_id", how="anti"
            )
            if unmatched.height:
                raise DataQualityError("Evaluation references an unknown prediction")
    return {
        "status": "VALID",
        "model_version": policy["model_version"],
        "predictions": predictions.height,
        "evaluations": evaluations.height,
        "decision_policy": "PASS-only",
    }


def validate_all(settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    duplicate_exports = sorted((resolved.data_dir / "curated").glob("*.csv"))
    if duplicate_exports:
        raise DataQualityError("Duplicate authoritative CSVs remain under data/curated")

    results: dict[str, Any] = {}
    loaded: dict[str, pl.DataFrame] = {}
    with connect(resolved) as connection:
        for name, schema in ARTIFACT_SCHEMAS.items():
            csv_path = resolved.root / f"{name}.csv"
            if not csv_path.exists():
                raise DataQualityError(f"Missing authoritative artifact: {csv_path.name}")
            frame = pl.read_csv(csv_path, infer_schema_length=100_000)
            loaded[name] = frame
            if tuple(frame.columns) != schema.columns:
                raise DataQualityError(
                    f"{name}.csv columns do not match schema version {resolved.schema_version}"
                )
            if table_columns(connection, name) != schema.columns:
                raise DataQualityError(f"SQLite table {name} does not match its CSV contract")
            duplicates = frame.group_by(schema.primary_key).len().filter(pl.col("len") > 1)
            if not duplicates.is_empty():
                raise DataQualityError(f"{name}.csv contains duplicate primary keys")
            for required_metadata in (
                "source",
                "retrieved_at_utc",
                "schema_version",
                "content_hash",
            ):
                if frame.height and frame[required_metadata].null_count():
                    raise DataQualityError(
                        f"{name}.csv has missing required metadata: {required_metadata}"
                    )
            _validate_casts(name, frame)
            _validate_hashes(name, frame)
            database_rows = int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            if database_rows != frame.height:
                raise DataQualityError(
                    f"{name}.csv has {frame.height} rows but SQLite has {database_rows}"
                )
            results[name] = {
                "rows": frame.height,
                "schema": "VALID",
                "duplicates": 0,
                "database_rows": database_rows,
            }

    games = loaded["games"]
    metrics = loaded["team_metrics"]
    if metrics.height:
        unmatched = metrics.join(games.select("game_id"), on="game_id", how="anti")
        if not unmatched.is_empty():
            raise DataQualityError("team_metrics.csv contains unmatched game_id values")
        counts = metrics.group_by("game_id").len().filter(pl.col("len") != 2)
        if not counts.is_empty():
            raise DataQualityError("team_metrics.csv must contain exactly two team rows per game")
        kickoff_check = metrics.join(
            games.select("game_id", pl.col("kickoff_utc").alias("game_kickoff_utc")),
            on="game_id",
            how="left",
            validate="m:1",
        ).filter(pl.col("kickoff_utc") != pl.col("game_kickoff_utc"))
        if not kickoff_check.is_empty():
            raise DataQualityError("team_metrics kickoff does not match canonical games kickoff")
    usage = loaded["player_usage"]
    if usage.height:
        unmatched_usage = usage.join(games.select("game_id"), on="game_id", how="anti")
        if not unmatched_usage.is_empty():
            raise DataQualityError("player_usage.csv contains unmatched game_id values")

    lag_path = resolved.runtime_dir / "features" / "team_game_lags.parquet"
    if lag_path.exists():
        lags = pl.read_parquet(lag_path)
        try:
            validate_lag_boundaries(lags)
        except ValueError as exc:
            raise DataQualityError(str(exc)) from exc
        results["leakage"] = {"status": "VALID", "rows": lags.height}
    else:
        results["leakage"] = {"status": "NOT_BUILT", "rows": 0}

    if games.height:
        sync_manifest = resolved.manifests_dir / "nflverse_sync.latest.json"
        if not sync_manifest.exists():
            raise DataQualityError(
                "games.csv is populated but the completed sync manifest is missing"
            )
        payload = json.loads(sync_manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise DataQualityError("Latest nflverse synchronization is not complete")
        results["sync_manifest"] = {
            "status": "VALID",
            "run_id": payload.get("run_id"),
            "retrieved_at_utc": payload.get("retrieved_at_utc"),
            "seasons": payload.get("seasons"),
        }
    coverage = _validate_model_coverage(games, metrics)
    results["model_coverage"] = coverage
    if coverage.get("excluded_games"):
        exclusions = pl.DataFrame(coverage["excluded_games"])
        atomic_write_text(resolved.reports_dir / "data_exclusions.csv", exclusions.write_csv())
    if coverage.get("breaches"):
        raise DataQualityError("; ".join(coverage["breaches"]))
    results["v11"] = _validate_v11(resolved)
    results["prospective"] = _validate_prospective(
        resolved,
        loaded["model_predictions"],
        loaded["prospective_evaluations"],
    )
    results["status"] = "VALID"
    return results
