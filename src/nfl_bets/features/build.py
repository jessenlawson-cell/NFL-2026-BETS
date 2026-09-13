from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import initialize_database, transaction
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_text, canonical_hash, iso_utc

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


def _latest_raw(settings: Settings, prefix: str) -> Path:
    candidates = sorted((settings.raw_dir / "nflverse").glob(f"{prefix}_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No cached {prefix} parquet found; run nfl-bets data sync first")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _game_level_observations(pbp: pl.DataFrame) -> pl.DataFrame:
    regular = pbp.filter(
        (pl.col("season_type") == "REG")
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
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
    home = base.select(
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        pl.col("home_team").alias("team_id"),
        pl.col("away_team").alias("opponent_team_id"),
        pl.lit(True).alias("is_home"),
    )
    away = base.select(
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        pl.col("away_team").alias("team_id"),
        pl.col("home_team").alias("opponent_team_id"),
        pl.lit(False).alias("is_home"),
    )
    schedule = pl.concat([home, away]).sort(["team_id", "kickoff_dt", "game_id"])
    completed = schedule.filter(pl.col("kickoff_dt") <= pl.lit(as_of))
    next_games = (
        schedule.filter(pl.col("kickoff_dt") > pl.lit(as_of))
        .group_by("team_id", maintain_order=True)
        .first()
    )
    return (
        pl.concat([completed, next_games], how="diagonal_relaxed")
        .unique(["game_id", "team_id"], keep="first")
        .sort(["team_id", "kickoff_dt", "game_id"])
    )


def _weighted_expression(metric: str, half_life: float, include_current: bool) -> pl.Expr:
    start = 0 if include_current else 1
    weighted_terms: list[pl.Expr] = []
    valid_weights: list[pl.Expr] = []
    for offset in range(start, start + 4):
        # Pregame features always begin at shift(1); this is the leakage boundary.
        shifted = pl.col(metric).shift(offset).over(["team_id", "season"])
        weight = 2.0 ** (-(offset - start) / half_life)
        weighted_terms.append(shifted.fill_null(0.0) * weight)
        valid_weights.append(shifted.is_not_null().cast(pl.Float64) * weight)
    numerator = sum(weighted_terms[1:], weighted_terms[0])
    denominator = sum(valid_weights[1:], valid_weights[0])
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def build_team_metric_matrix(team_games: pl.DataFrame, half_life: float = 2.0) -> pl.DataFrame:
    """Build past-only four-game EWMAs; shift(1) is applied before every rolling term."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    required = {"team_id", "season", "kickoff_dt", *METRICS}
    missing = required - set(team_games.columns)
    if missing:
        raise ValueError(f"Feature input missing columns: {sorted(missing)}")
    ordered = team_games.sort(["team_id", "season", "kickoff_dt", "game_id"])
    ewma_columns = [
        _weighted_expression(metric, half_life, include_current=False).alias(f"current_{metric}")
        for metric in METRICS
    ]
    shifted_count_terms = [
        pl.col(METRICS[0]).shift(offset).over(["team_id", "season"]).is_not_null().cast(pl.Int32)
        for offset in range(1, 5)
    ]
    games_available = sum(shifted_count_terms[1:], shifted_count_terms[0]).alias("games_available")
    current = ordered.with_columns(*ewma_columns, games_available)

    completed = ordered.filter(pl.col(METRICS[0]).is_not_null()).with_columns(
        *[
            _weighted_expression(metric, half_life, include_current=True).alias(f"prior_{metric}")
            for metric in METRICS
        ]
    )
    priors = (
        completed.group_by(["team_id", "season"], maintain_order=True)
        .last()
        .select(
            "team_id",
            (pl.col("season") + 1).alias("season"),
            *[f"prior_{metric}" for metric in METRICS],
        )
    )
    blended = current.join(priors, on=["team_id", "season"], how="left", validate="m:1")
    current_weight = pl.col("games_available").clip(0, 4).cast(pl.Float64) / 4.0
    blend_columns = []
    for metric in METRICS:
        current_value = pl.col(f"current_{metric}")
        prior_value = pl.col(f"prior_{metric}")
        blend_columns.append(
            pl.when(current_value.is_null())
            .then(prior_value)
            .when(prior_value.is_null())
            .then(current_value)
            .otherwise(current_weight * current_value + (1.0 - current_weight) * prior_value)
            .alias(metric)
        )
    return blended.with_columns(*blend_columns).drop(
        *[f"current_{metric}" for metric in METRICS],
        *[f"prior_{metric}" for metric in METRICS],
    )


def _upsert_team_metrics(records: list[dict[str, object]], settings: Settings) -> None:
    if not records:
        return
    columns = ARTIFACT_SCHEMAS["team_metrics"].columns
    update_columns = [column for column in columns if column not in {"game_id", "team_id"}]
    with transaction(settings) as connection:
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in update_columns)
        connection.executemany(
            f"INSERT INTO team_metrics ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(game_id,team_id) DO UPDATE SET {updates}",
            [tuple(record[column] for column in columns) for record in records],
        )


def build_features(
    as_of: datetime,
    half_life: float = 2.0,
    settings: Settings | None = None,
) -> dict[str, object]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    if as_of.tzinfo is None:
        raise ValueError("--as-of must include a timezone")
    as_of_utc = as_of.astimezone(UTC)
    pbp = pl.read_parquet(_latest_raw(resolved, "pbp"))
    games = pl.read_csv(resolved.root / "games.csv", infer_schema_length=10_000)
    observations = _game_level_observations(pbp)
    schedule = _team_schedule(games, as_of_utc)
    team_games = schedule.join(
        observations,
        on=["game_id", "season", "week", "team_id"],
        how="left",
        validate="1:1",
    )
    matrix = build_team_metric_matrix(team_games, half_life)
    retrieved_at = iso_utc()
    feature_as_of = iso_utc(as_of_utc)
    records: list[dict[str, object]] = []
    for row in matrix.iter_rows(named=True):
        record = {
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "kickoff_utc": iso_utc(row["kickoff_dt"]),
            "team_id": row["team_id"],
            "opponent_team_id": row["opponent_team_id"],
            "is_home": int(row["is_home"]),
            "games_available": row["games_available"],
            **{metric: row[metric] for metric in METRICS},
            "feature_as_of_utc": feature_as_of,
            "half_life": half_life,
            "source": "nflverse:pbp-derived",
            "retrieved_at_utc": retrieved_at,
            "source_updated_at_utc": None,
            "schema_version": resolved.schema_version,
        }
        record["content_hash"] = canonical_hash(record)
        records.append(record)
    _upsert_team_metrics(records, resolved)
    columns = ARTIFACT_SCHEMAS["team_metrics"].columns
    frame = (
        pl.DataFrame(records).select(columns) if records else pl.DataFrame({c: [] for c in columns})
    )
    atomic_write_text(resolved.root / "team_metrics.csv", frame.write_csv())
    atomic_write_text(resolved.curated_dir / "team_metrics.csv", frame.write_csv())
    manifest = {
        "artifact": "team_metrics.csv",
        "as_of_utc": feature_as_of,
        "rows": len(records),
        "half_life": half_life,
        "rolling_games": 4,
        "lag": 1,
        "source": "nflverse:pbp-derived",
        "schema_version": resolved.schema_version,
    }
    manifest_path = resolved.curated_dir / "team_metrics.manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, sort_keys=True, indent=2))
    return manifest
