from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.features.build import (
    METRICS,
    _game_level_observations,
    _load_sync_manifest,
    _team_schedule,
    _write_parquet_atomic,
    build_lag_matrix,
    validate_lag_boundaries,
)
from nfl_bets.teams import canonical_team_column
from nfl_bets.util import atomic_write_text, iso_utc, sha256_bytes

PRESSURE_METRICS = (
    "off_sack_rate",
    "def_sack_rate",
    "off_qb_hit_rate",
    "def_qb_hit_rate",
)
NEUTRAL_RUSH_METRICS = (
    "off_neutral_rush_epa",
    "def_neutral_rush_epa",
    "off_neutral_rush_success_rate",
    "def_neutral_rush_success_rate",
)
CONTINUITY_METRICS = ("off_snap_continuity", "def_snap_continuity")
V11_METRICS = (*METRICS, *PRESSURE_METRICS, *NEUTRAL_RUSH_METRICS, *CONTINUITY_METRICS)
INJURY_DIAGNOSTICS = (
    "qb_out_doubtful",
    "ol_out_doubtful",
    "defensive_front_out_doubtful",
    "secondary_out_doubtful",
    "questionable_players",
)
V11_START_SEASON = 2013


def _pressure_and_neutral_observations(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "epa",
        "success",
        "qb_dropback",
        "rush_attempt",
        "sack",
        "qb_hit",
        "qtr",
        "score_differential",
    }
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"V1.1 play-by-play input missing columns: {sorted(missing)}")
    season_type = pl.col("season_type") == "REG" if "season_type" in pbp.columns else pl.lit(True)
    regular = pbp.filter(
        season_type
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
    ).with_columns(canonical_team_column("posteam"), canonical_team_column("defteam"))
    kneel = pl.col("qb_kneel").fill_null(0) if "qb_kneel" in regular.columns else pl.lit(0)
    dropbacks = regular.filter(pl.col("qb_dropback") == 1)
    neutral_rushes = regular.filter(
        (pl.col("rush_attempt") == 1)
        & (kneel != 1)
        & (pl.col("qtr") <= 3)
        & (pl.col("score_differential").abs() <= 8)
    )
    keys = ["game_id", "season", "week"]
    offense_pressure = (
        dropbacks.group_by([*keys, "posteam"])
        .agg(
            pl.col("sack").mean().alias("off_sack_rate"),
            pl.col("qb_hit").mean().alias("off_qb_hit_rate"),
        )
        .rename({"posteam": "team_id"})
    )
    defense_pressure = (
        dropbacks.group_by([*keys, "defteam"])
        .agg(
            pl.col("sack").mean().alias("def_sack_rate"),
            pl.col("qb_hit").mean().alias("def_qb_hit_rate"),
        )
        .rename({"defteam": "team_id"})
    )
    offense_neutral = (
        neutral_rushes.group_by([*keys, "posteam"])
        .agg(
            pl.col("epa").mean().alias("off_neutral_rush_epa"),
            pl.col("success").mean().alias("off_neutral_rush_success_rate"),
        )
        .rename({"posteam": "team_id"})
    )
    defense_neutral = (
        neutral_rushes.group_by([*keys, "defteam"])
        .agg(
            pl.col("epa").mean().alias("def_neutral_rush_epa"),
            pl.col("success").mean().alias("def_neutral_rush_success_rate"),
        )
        .rename({"defteam": "team_id"})
    )
    join_keys = [*keys, "team_id"]
    return (
        _game_level_observations(pbp)
        .join(offense_pressure, on=join_keys, how="left", validate="1:1")
        .join(defense_pressure, on=join_keys, how="left", validate="1:1")
        .join(offense_neutral, on=join_keys, how="left", validate="1:1")
        .join(defense_neutral, on=join_keys, how="left", validate="1:1")
    )


def _snap_continuity(usage: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "player_id",
        "team_id",
        "offense_pct",
        "defense_pct",
    }
    missing = required - set(usage.columns)
    if missing:
        raise ValueError(f"Player usage input missing columns: {sorted(missing)}")
    regular = games.filter(pl.col("game_type") == "REG").with_columns(
        pl.col("kickoff_utc").str.to_datetime(time_zone="UTC", strict=False).alias("kickoff_dt")
    )
    home = regular.select(
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        pl.col("home_team").alias("team_id"),
    )
    away = regular.select(
        "game_id",
        "season",
        "week",
        "kickoff_dt",
        pl.col("away_team").alias("team_id"),
    )
    order = (
        pl.concat([home, away])
        .join(
            usage.select("game_id", "team_id").unique(),
            on=["game_id", "team_id"],
            how="inner",
            validate="1:1",
        )
        .sort(["team_id", "season", "kickoff_dt", "game_id"])
        .with_columns(
            pl.col("game_id").cum_count().over(["team_id", "season"]).alias("game_number")
        )
    )
    current = usage.select(
        "season",
        "week",
        "game_id",
        "player_id",
        "team_id",
        pl.col("offense_pct").cast(pl.Float64).fill_null(0.0),
        pl.col("defense_pct").cast(pl.Float64).fill_null(0.0),
    ).join(order, on=["season", "week", "game_id", "team_id"], validate="m:1")
    previous = current.select(
        "season",
        "team_id",
        "player_id",
        (pl.col("game_number") + 1).alias("game_number"),
        pl.col("offense_pct").alias("previous_offense_pct"),
        pl.col("defense_pct").alias("previous_defense_pct"),
    )
    paired = current.join(
        previous,
        on=["season", "team_id", "player_id", "game_number"],
        how="left",
        validate="m:1",
    )
    return paired.group_by(
        ["game_id", "season", "week", "team_id", "game_number"], maintain_order=True
    ).agg(
        pl.col("offense_pct").sum().alias("offense_denominator"),
        pl.min_horizontal(
            pl.col("offense_pct"), pl.col("previous_offense_pct").fill_null(0.0)
        )
        .sum()
        .alias("offense_overlap"),
        pl.col("defense_pct").sum().alias("defense_denominator"),
        pl.min_horizontal(
            pl.col("defense_pct"), pl.col("previous_defense_pct").fill_null(0.0)
        )
        .sum()
        .alias("defense_overlap"),
    ).with_columns(
        pl.when((pl.col("game_number") > 1) & (pl.col("offense_denominator") > 0))
        .then(pl.col("offense_overlap") / pl.col("offense_denominator"))
        .otherwise(None)
        .alias("off_snap_continuity"),
        pl.when((pl.col("game_number") > 1) & (pl.col("defense_denominator") > 0))
        .then(pl.col("defense_overlap") / pl.col("defense_denominator"))
        .otherwise(None)
        .alias("def_snap_continuity"),
    ).select("game_id", "season", "week", "team_id", *CONTINUITY_METRICS)


def _injury_diagnostics(injuries: pl.DataFrame) -> pl.DataFrame:
    if injuries.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "team_id": pl.String,
                **{metric: pl.Int64 for metric in INJURY_DIAGNOSTICS},
            }
        )
    severe = pl.col("injury_status").is_in(["Out", "Doubtful"])
    return injuries.with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
    ).group_by(["season", "week", "team_id"]).agg(
        (severe & (pl.col("position") == "QB")).sum().alias("qb_out_doubtful"),
        (severe & pl.col("position").is_in(["C", "G", "T"]))
        .sum()
        .alias("ol_out_doubtful"),
        (severe & pl.col("position").is_in(["DE", "DT", "LB"]))
        .sum()
        .alias("defensive_front_out_doubtful"),
        (severe & pl.col("position").is_in(["CB", "S"]))
        .sum()
        .alias("secondary_out_doubtful"),
        (pl.col("injury_status") == "Questionable").sum().alias("questionable_players"),
    )


def build_v11_features(
    as_of: datetime, settings: Settings | None = None
) -> dict[str, Any]:
    resolved = settings or get_settings()
    resolved.ensure_directories()
    if as_of.tzinfo is None:
        raise ValueError("--as-of must include a timezone")
    as_of_utc = as_of.astimezone(UTC)
    sync_manifest = _load_sync_manifest(resolved)
    sync_as_of = datetime.fromisoformat(
        str(sync_manifest["retrieved_at_utc"]).replace("Z", "+00:00")
    ).astimezone(UTC)
    if as_of_utc != sync_as_of:
        raise ValueError(
            "Production V1.1 features must use the completed synchronization retrieval time"
        )
    pbp_entries = [
        entry
        for entry in sync_manifest.get("raw_datasets", {}).values()
        if entry.get("dataset") == "pbp" and int(entry["season"]) >= V11_START_SEASON
    ]
    if not pbp_entries:
        raise FileNotFoundError("Completed sync manifest has no V1.1 play-by-play history")
    observations: list[pl.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for entry in sorted(pbp_entries, key=lambda item: int(item["season"])):
        path = resolved.root / str(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Manifest raw file is missing: {entry['path']}")
        source_hashes[f"pbp:{entry['season']}"] = str(entry["content_hash"])
        observations.append(_pressure_and_neutral_observations(pl.read_parquet(path)))
    observation_frame = pl.concat(observations, how="diagonal_relaxed")
    games = pl.read_csv(resolved.root / "games.csv", infer_schema_length=100_000)
    schedule = _team_schedule(games, as_of_utc).filter(pl.col("season") >= V11_START_SEASON)
    usage = pl.read_csv(resolved.root / "player_usage.csv", infer_schema_length=100_000)
    source_hashes["player_usage.csv"] = sha256_bytes(
        (resolved.root / "player_usage.csv").read_bytes()
    )
    continuity = _snap_continuity(usage, games)
    team_games = (
        schedule.join(
            observation_frame,
            on=["game_id", "season", "week", "team_id"],
            how="left",
            validate="1:1",
        )
        .join(
            continuity,
            on=["game_id", "season", "week", "team_id"],
            how="left",
            validate="1:1",
        )
    )
    lags = build_lag_matrix(team_games, V11_METRICS)
    validate_lag_boundaries(lags)
    if lags.select("game_id", "team_id").is_duplicated().any():
        raise ValueError("Duplicate V1.1 team-game feature keys detected")
    incomplete_games = lags.group_by("game_id").len().filter(pl.col("len") != 2)
    if incomplete_games.height:
        raise ValueError("V1.1 feature store does not contain exactly two teams per game")
    injuries = pl.read_csv(resolved.root / "injuries.csv", infer_schema_length=100_000)
    source_hashes["injuries.csv"] = sha256_bytes(
        (resolved.root / "injuries.csv").read_bytes()
    )
    injury = _injury_diagnostics(injuries)
    lags = lags.join(
        injury,
        on=["season", "week", "team_id"],
        how="left",
        validate="m:1",
    )
    feature_dir = resolved.runtime_dir / "features"
    lag_path = feature_dir / "v11_team_game_lags.parquet"
    _write_parquet_atomic(lags, lag_path)
    injury_seasons = sorted(
        int(value) for value in injuries["season"].drop_nulls().unique().to_list()
    )
    manifest = {
        "artifact": "data/runtime/features/v11_team_game_lags.parquet",
        "status": "BUILT_UNWEIGHTED",
        "as_of_utc": iso_utc(as_of_utc),
        "retrieved_at_utc": iso_utc(),
        "rows": lags.height,
        "seasons": sorted(int(value) for value in lags["season"].unique().to_list()),
        "rolling_games": 4,
        "lags": [1, 2, 3, 4],
        "metrics": list(V11_METRICS),
        "neutral_rush_definition": (
            "REG plays; rush_attempt=1; qb_kneel!=1; quarters 1-3; "
            "absolute pre-play score differential <=8"
        ),
        "snap_continuity_definition": (
            "current player snap-share overlap with the immediately previous team game, "
            "divided by current unit snap-share"
        ),
        "injury_diagnostics": list(INJURY_DIAGNOSTICS),
        "injury_seasons": injury_seasons,
        "injury_model_status": "DISABLED_INSUFFICIENT_HISTORICAL_COVERAGE",
        "source": "nflverse:pbp,snap-counts,injuries",
        "source_hashes": source_hashes,
        "sync_run_id": sync_manifest["run_id"],
        "schema_version": resolved.schema_version,
        "content_hash": sha256_bytes(lag_path.read_bytes()),
    }
    atomic_write_text(
        resolved.manifests_dir / "v11_feature_inputs.latest.json",
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    return manifest
