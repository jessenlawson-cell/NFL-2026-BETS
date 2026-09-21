from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.teams import canonical_team_column
from nfl_bets.util import atomic_write_text, iso_utc, sha256_bytes

METRICS = (
    "off_pass_epa",
    "def_pass_epa",
    "off_rush_epa",
    "def_rush_epa",
    "off_pass_success_rate",
    "def_pass_success_rate",
    "off_rush_success_rate",
    "def_rush_success_rate",
)
HALF_LIFE_GRID = (1.0, 2.0, 4.0)
PRIOR_STRENGTH_GRID = (0.25, 0.5, 0.75, 1.0)
LAG_COUNT = 4


def _write_parquet_atomic(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    try:
        frame.write_parquet(temporary_name, compression="zstd")
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _game_level_observations(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {"game_id", "season", "week", "posteam", "defteam", "epa", "success"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Play-by-play input missing columns: {sorted(missing)}")
    season_type = pl.col("season_type") == "REG" if "season_type" in pbp.columns else pl.lit(True)
    regular = pbp.filter(
        season_type
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
    ).with_columns(
        canonical_team_column("posteam"),
        canonical_team_column("defteam"),
    )
    kneel = pl.col("qb_kneel").fill_null(0) if "qb_kneel" in regular.columns else pl.lit(0)
    pass_plays = regular.filter(pl.col("qb_dropback") == 1)
    rush_plays = regular.filter((pl.col("rush_attempt") == 1) & (kneel != 1))
    offense_pass = (
        pass_plays.group_by(["game_id", "season", "week", "posteam"])
        .agg(
            pl.col("epa").mean().alias("off_pass_epa"),
            pl.col("success").mean().alias("off_pass_success_rate"),
        )
        .rename({"posteam": "team_id"})
    )
    offense_rush = (
        rush_plays.group_by(["game_id", "season", "week", "posteam"])
        .agg(
            pl.col("epa").mean().alias("off_rush_epa"),
            pl.col("success").mean().alias("off_rush_success_rate"),
        )
        .rename({"posteam": "team_id"})
    )
    defense_pass = (
        pass_plays.group_by(["game_id", "season", "week", "defteam"])
        .agg(
            pl.col("epa").mean().alias("def_pass_epa"),
            pl.col("success").mean().alias("def_pass_success_rate"),
        )
        .rename({"defteam": "team_id"})
    )
    defense_rush = (
        rush_plays.group_by(["game_id", "season", "week", "defteam"])
        .agg(
            pl.col("epa").mean().alias("def_rush_epa"),
            pl.col("success").mean().alias("def_rush_success_rate"),
        )
        .rename({"defteam": "team_id"})
    )
    keys = ["game_id", "season", "week", "team_id"]
    return (
        offense_pass.join(offense_rush, on=keys, how="full", coalesce=True, validate="1:1")
        .join(defense_pass, on=keys, how="full", coalesce=True, validate="1:1")
        .join(defense_rush, on=keys, how="full", coalesce=True, validate="1:1")
    )


def _team_schedule(games: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    base = games.filter(pl.col("game_type") == "REG").with_columns(
        pl.col("kickoff_utc").str.to_datetime(time_zone="UTC", strict=False).alias("kickoff_dt")
    )
    if base["kickoff_dt"].null_count():
        raise ValueError("games.csv contains an invalid kickoff_utc value")
    completed = (
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        if {"home_score", "away_score"}.issubset(base.columns)
        else pl.lit(True)
    )
    home = base.select(
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        completed.alias("is_completed"),
        pl.col("home_team").alias("team_id"),
        pl.col("away_team").alias("opponent_team_id"),
        pl.lit(True).alias("is_home"),
    )
    away = base.select(
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        completed.alias("is_completed"),
        pl.col("away_team").alias("team_id"),
        pl.col("home_team").alias("opponent_team_id"),
        pl.lit(False).alias("is_home"),
    )
    schedule = pl.concat([home, away]).sort(["team_id", "kickoff_dt", "game_id"])
    history = schedule.filter(
        (pl.col("kickoff_dt") < pl.lit(as_of)) & pl.col("is_completed")
    )
    next_games = (
        schedule.filter(pl.col("kickoff_dt") > pl.lit(as_of))
        .group_by("team_id", maintain_order=True)
        .first()
    )
    selected = (
        pl.concat([history, next_games], how="diagonal_relaxed")
        .unique(["game_id", "team_id"], keep="first")
        .sort(["team_id", "kickoff_dt", "game_id"])
    )
    complete_games = selected.group_by("game_id").len().filter(pl.col("len") == 2).select(
        "game_id"
    )
    return selected.join(complete_games, on="game_id", how="inner", validate="m:1").drop(
        "is_completed"
    )


def build_lag_matrix(
    team_games: pl.DataFrame, metrics: tuple[str, ...] = METRICS
) -> pl.DataFrame:
    """Materialize explicit lag-1 through lag-4 values before any weighting."""
    required = {"game_id", "team_id", "season", "kickoff_dt", *metrics}
    missing = required - set(team_games.columns)
    if missing:
        raise ValueError(f"Feature input missing columns: {sorted(missing)}")
    ordered = team_games.sort(["team_id", "season", "kickoff_dt", "game_id"])
    expressions: list[pl.Expr] = []
    for offset in range(1, LAG_COUNT + 1):
        expressions.append(
            pl.col("kickoff_dt")
            .shift(offset)
            .over(["team_id", "season"])
            .alias(f"lag{offset}_kickoff_dt")
        )
        for metric in metrics:
            expressions.append(
                pl.col(metric)
                .shift(offset)
                .over(["team_id", "season"])
                .alias(f"lag{offset}_{metric}")
            )
    available = [
        pl.col(f"lag{offset}_{metrics[0]}").is_not_null().cast(pl.Int32)
        for offset in range(1, LAG_COUNT + 1)
    ]
    return ordered.with_columns(*expressions).with_columns(
        sum(available[1:], available[0]).alias("games_available")
    )


def _prior_frame(
    lags: pl.DataFrame, half_life: float, metrics: tuple[str, ...] = METRICS
) -> pl.DataFrame:
    ordered = lags.sort(["team_id", "season", "kickoff_dt", "game_id"])
    completed = ordered.filter(pl.col(metrics[0]).is_not_null())
    weighted: list[pl.Expr] = []
    for metric in metrics:
        numerator: pl.Expr | None = None
        denominator: pl.Expr | None = None
        for offset in range(0, LAG_COUNT):
            shifted = pl.col(metric).shift(offset).over(["team_id", "season"])
            weight = 2.0 ** (-offset / half_life)
            term = shifted.fill_null(0.0) * weight
            valid = shifted.is_not_null().cast(pl.Float64) * weight
            numerator = term if numerator is None else numerator + term
            denominator = valid if denominator is None else denominator + valid
        assert numerator is not None and denominator is not None
        weighted.append(
            pl.when(denominator > 0)
            .then(numerator / denominator)
            .otherwise(None)
            .alias(f"team_prior_{metric}")
        )
    team_prior = completed.with_columns(*weighted).group_by(
        ["team_id", "season"], maintain_order=True
    ).last()
    league = team_prior.group_by("season").agg(
        *[
            pl.col(f"team_prior_{metric}").mean().alias(f"league_prior_{metric}")
            for metric in metrics
        ]
    )
    return (
        team_prior.join(league, on="season", validate="m:1")
        .select(
            "team_id",
            (pl.col("season") + 1).alias("season"),
            *[f"team_prior_{metric}" for metric in metrics],
            *[f"league_prior_{metric}" for metric in metrics],
        )
    )


def materialize_feature_configuration(
    lags: pl.DataFrame,
    half_life: float,
    prior_strength: float,
    metrics: tuple[str, ...] = METRICS,
) -> pl.DataFrame:
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    if not 0.0 <= prior_strength <= 1.0:
        raise ValueError("prior_strength must be between zero and one")
    required_lags = {
        f"lag{offset}_{metric}" for offset in range(1, LAG_COUNT + 1) for metric in metrics
    }
    missing = required_lags - set(lags.columns)
    if missing:
        raise ValueError(f"Lag store missing columns: {sorted(missing)}")
    priors = _prior_frame(lags, half_life, metrics)
    joined = lags.join(priors, on=["team_id", "season"], how="left", validate="m:1")
    current_weight = pl.col("games_available").clip(0, LAG_COUNT).cast(pl.Float64) / LAG_COUNT
    expressions: list[pl.Expr] = []
    for metric in metrics:
        numerator: pl.Expr | None = None
        denominator: pl.Expr | None = None
        for offset in range(1, LAG_COUNT + 1):
            lagged = pl.col(f"lag{offset}_{metric}")
            weight = 2.0 ** (-(offset - 1) / half_life)
            term = lagged.fill_null(0.0) * weight
            valid = lagged.is_not_null().cast(pl.Float64) * weight
            numerator = term if numerator is None else numerator + term
            denominator = valid if denominator is None else denominator + valid
        assert numerator is not None and denominator is not None
        current = pl.when(denominator > 0).then(numerator / denominator).otherwise(None)
        prior = (
            prior_strength * pl.col(f"team_prior_{metric}")
            + (1.0 - prior_strength) * pl.col(f"league_prior_{metric}")
        )
        expressions.append(
            pl.when(current.is_null())
            .then(prior)
            .when(prior.is_null())
            .then(current)
            .otherwise(current_weight * current + (1.0 - current_weight) * prior)
            .alias(metric)
        )
    return joined.with_columns(*expressions).drop(
        *[f"team_prior_{metric}" for metric in metrics],
        *[f"league_prior_{metric}" for metric in metrics],
    )


def build_team_metric_matrix(
    team_games: pl.DataFrame, half_life: float = 2.0, prior_strength: float = 1.0
) -> pl.DataFrame:
    """Compatibility helper for a leakage-safe weighted team-game matrix."""
    return materialize_feature_configuration(
        build_lag_matrix(team_games), half_life=half_life, prior_strength=prior_strength
    )


def validate_lag_boundaries(lags: pl.DataFrame) -> None:
    violations: list[str] = []
    for offset in range(1, LAG_COUNT + 1):
        column = f"lag{offset}_kickoff_dt"
        invalid = lags.filter(
            pl.col(column).is_not_null() & (pl.col(column) >= pl.col("kickoff_dt"))
        )
        if invalid.height:
            violations.append(f"lag {offset}: {invalid.height}")
    if violations:
        raise ValueError("Future or same-kickoff feature inputs detected: " + ", ".join(violations))


def _load_sync_manifest(settings: Settings) -> dict[str, Any]:
    path = settings.manifests_dir / "nflverse_sync.latest.json"
    if not path.exists():
        raise FileNotFoundError("No completed sync manifest found; run nfl-bets data sync first")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or not payload.get("include_pbp"):
        raise ValueError("Latest synchronization is not a complete play-by-play run")
    return payload


def build_features(as_of: datetime, settings: Settings | None = None) -> dict[str, object]:
    resolved = settings or get_settings()
    resolved.ensure_directories()
    if as_of.tzinfo is None:
        raise ValueError("--as-of must include a timezone")
    as_of_utc = as_of.astimezone(UTC)
    sync_manifest = _load_sync_manifest(resolved)
    pbp_entries = [
        entry
        for entry in sync_manifest.get("raw_datasets", {}).values()
        if entry.get("dataset") == "pbp"
    ]
    if not pbp_entries:
        raise FileNotFoundError("Completed sync manifest does not contain play-by-play data")
    observations: list[pl.DataFrame] = []
    for entry in sorted(pbp_entries, key=lambda item: int(item["season"])):
        path = resolved.root / str(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Manifest raw file is missing: {entry['path']}")
        observations.append(_game_level_observations(pl.read_parquet(path)))
    observation_frame = pl.concat(observations, how="diagonal_relaxed")
    games = pl.read_csv(resolved.root / "games.csv", infer_schema_length=100_000)
    schedule = _team_schedule(games, as_of_utc)
    team_games = schedule.join(
        observation_frame,
        on=["game_id", "season", "week", "team_id"],
        how="left",
        validate="1:1",
    )
    lags = build_lag_matrix(team_games)
    validate_lag_boundaries(lags)
    feature_dir = resolved.runtime_dir / "features"
    lag_path = feature_dir / "team_game_lags.parquet"
    _write_parquet_atomic(lags, lag_path)
    content_hash = sha256_bytes(lag_path.read_bytes())
    manifest = {
        "artifact": "data/runtime/features/team_game_lags.parquet",
        "status": "BUILT_UNWEIGHTED",
        "as_of_utc": iso_utc(as_of_utc),
        "retrieved_at_utc": iso_utc(),
        "rows": lags.height,
        "rolling_games": LAG_COUNT,
        "lags": [1, 2, 3, 4],
        "half_life_grid": list(HALF_LIFE_GRID),
        "prior_strength_grid": list(PRIOR_STRENGTH_GRID),
        "source": "nflverse:pbp-derived",
        "sync_run_id": sync_manifest["run_id"],
        "schema_version": resolved.schema_version,
        "content_hash": content_hash,
    }
    atomic_write_text(
        resolved.manifests_dir / "team_feature_inputs.latest.json",
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    return manifest
