from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Final

import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.features.build import (
    _load_sync_manifest,
    _team_schedule,
    _write_parquet_atomic,
    build_lag_matrix,
    validate_lag_boundaries,
)
from nfl_bets.features.v11 import (
    CONTINUITY_METRICS,
    INJURY_DIAGNOSTICS,
    _injury_diagnostics,
    _snap_continuity,
)
from nfl_bets.teams import canonical_team_column
from nfl_bets.util import atomic_write_text, iso_utc, sha256_bytes

KEYS: Final = ("game_id", "season", "week")

PASS_METRICS: Final = (
    "off_wp_pass_epa",
    "def_wp_pass_epa",
    "off_wp_pass_success_rate",
    "def_wp_pass_success_rate",
    "off_cpoe",
    "def_cpoe",
    "off_air_epa",
    "def_air_epa",
    "off_yac_epa",
    "def_yac_epa",
    "off_pass_explosive_rate",
    "def_pass_explosive_rate",
)
RUSH_METRICS: Final = (
    "off_wp_rush_epa",
    "def_wp_rush_epa",
    "off_wp_rush_success_rate",
    "def_wp_rush_success_rate",
    "off_rush_explosive_rate",
    "def_rush_explosive_rate",
    "off_short_yard_success_rate",
    "def_short_yard_success_rate",
)
TRENCH_METRICS: Final = (
    "off_sack_rate",
    "def_sack_rate",
    "off_qb_hit_rate",
    "def_qb_hit_rate",
)
TENDENCY_METRICS: Final = (
    "off_neutral_proe",
    "def_neutral_proe_allowed",
    "off_early_down_pass_rate",
    "def_early_down_pass_rate_allowed",
    "off_no_huddle_rate",
    "def_no_huddle_rate_allowed",
    "off_shotgun_rate",
    "def_shotgun_rate_allowed",
)
SPECIAL_TEAMS_METRICS: Final = (
    "st_epa",
    "st_success_rate",
    "st_field_goal_epa",
    "st_extra_point_epa",
    "st_punt_epa",
    "st_kickoff_epa",
)
CHALLENGER_PLAY_METRICS: Final = (
    *PASS_METRICS,
    *RUSH_METRICS,
    *TRENCH_METRICS,
    *TENDENCY_METRICS,
    *SPECIAL_TEAMS_METRICS,
)
CHALLENGER_METRICS: Final = (*CHALLENGER_PLAY_METRICS, *CONTINUITY_METRICS)
CHALLENGER_START_SEASON: Final = 2021


def _rename_side(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    return frame.rename({source: "team_id"})


def _advanced_observations(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "epa",
        "success",
        "wp",
        "pass_oe",
        "qb_dropback",
        "rush_attempt",
        "sack",
        "qb_hit",
        "cpoe",
        "air_epa",
        "yac_epa",
        "yards_gained",
        "down",
        "ydstogo",
        "no_huddle",
        "shotgun",
        "special_teams_play",
        "play_type",
    }
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Challenger play-by-play input missing columns: {sorted(missing)}")

    season_type = pl.col("season_type") == "REG" if "season_type" in pbp.columns else pl.lit(True)
    regular = pbp.filter(
        season_type
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
    ).with_columns(canonical_team_column("posteam"), canonical_team_column("defteam"))
    neutral = regular.filter(pl.col("wp").is_between(0.05, 0.95, closed="none"))
    kneel = pl.col("qb_kneel").fill_null(0) if "qb_kneel" in neutral.columns else pl.lit(0)
    dropbacks = neutral.filter(pl.col("qb_dropback") == 1)
    rushes = neutral.filter((pl.col("rush_attempt") == 1) & (kneel != 1))
    scrimmage = neutral.filter((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
    all_dropbacks = regular.filter(pl.col("qb_dropback") == 1)
    short_yardage = rushes.filter(pl.col("down").is_in([3, 4]) & (pl.col("ydstogo") <= 2))
    special = regular.filter(pl.col("special_teams_play") == 1)

    base = pl.concat(
        [
            regular.select(*KEYS, pl.col("posteam").alias("team_id")),
            regular.select(*KEYS, pl.col("defteam").alias("team_id")),
        ]
    ).unique([*KEYS, "team_id"])

    offense_pass = _rename_side(
        dropbacks.group_by([*KEYS, "posteam"]).agg(
            pl.col("epa").mean().alias("off_wp_pass_epa"),
            pl.col("success").mean().alias("off_wp_pass_success_rate"),
            pl.col("cpoe").mean().alias("off_cpoe"),
            pl.col("air_epa").mean().alias("off_air_epa"),
            pl.col("yac_epa").mean().alias("off_yac_epa"),
            (pl.col("yards_gained") >= 15).mean().alias("off_pass_explosive_rate"),
        ),
        "posteam",
    )
    defense_pass = _rename_side(
        dropbacks.group_by([*KEYS, "defteam"]).agg(
            pl.col("epa").mean().alias("def_wp_pass_epa"),
            pl.col("success").mean().alias("def_wp_pass_success_rate"),
            pl.col("cpoe").mean().alias("def_cpoe"),
            pl.col("air_epa").mean().alias("def_air_epa"),
            pl.col("yac_epa").mean().alias("def_yac_epa"),
            (pl.col("yards_gained") >= 15).mean().alias("def_pass_explosive_rate"),
        ),
        "defteam",
    )
    offense_rush = _rename_side(
        rushes.group_by([*KEYS, "posteam"]).agg(
            pl.col("epa").mean().alias("off_wp_rush_epa"),
            pl.col("success").mean().alias("off_wp_rush_success_rate"),
            (pl.col("yards_gained") >= 10).mean().alias("off_rush_explosive_rate"),
        ),
        "posteam",
    )
    defense_rush = _rename_side(
        rushes.group_by([*KEYS, "defteam"]).agg(
            pl.col("epa").mean().alias("def_wp_rush_epa"),
            pl.col("success").mean().alias("def_wp_rush_success_rate"),
            (pl.col("yards_gained") >= 10).mean().alias("def_rush_explosive_rate"),
        ),
        "defteam",
    )
    offense_short = _rename_side(
        short_yardage.group_by([*KEYS, "posteam"]).agg(
            pl.col("success").mean().alias("off_short_yard_success_rate")
        ),
        "posteam",
    )
    defense_short = _rename_side(
        short_yardage.group_by([*KEYS, "defteam"]).agg(
            pl.col("success").mean().alias("def_short_yard_success_rate")
        ),
        "defteam",
    )
    offense_pressure = _rename_side(
        all_dropbacks.group_by([*KEYS, "posteam"]).agg(
            pl.col("sack").mean().alias("off_sack_rate"),
            pl.col("qb_hit").mean().alias("off_qb_hit_rate"),
        ),
        "posteam",
    )
    defense_pressure = _rename_side(
        all_dropbacks.group_by([*KEYS, "defteam"]).agg(
            pl.col("sack").mean().alias("def_sack_rate"),
            pl.col("qb_hit").mean().alias("def_qb_hit_rate"),
        ),
        "defteam",
    )
    offense_tendency = _rename_side(
        scrimmage.group_by([*KEYS, "posteam"]).agg(
            pl.col("pass_oe").mean().alias("off_neutral_proe"),
            pl.when(pl.col("down") <= 2)
            .then(pl.col("qb_dropback"))
            .mean()
            .alias("off_early_down_pass_rate"),
            pl.col("no_huddle").mean().alias("off_no_huddle_rate"),
            pl.col("shotgun").mean().alias("off_shotgun_rate"),
        ),
        "posteam",
    )
    defense_tendency = _rename_side(
        scrimmage.group_by([*KEYS, "defteam"]).agg(
            pl.col("pass_oe").mean().alias("def_neutral_proe_allowed"),
            pl.when(pl.col("down") <= 2)
            .then(pl.col("qb_dropback"))
            .mean()
            .alias("def_early_down_pass_rate_allowed"),
            pl.col("no_huddle").mean().alias("def_no_huddle_rate_allowed"),
            pl.col("shotgun").mean().alias("def_shotgun_rate_allowed"),
        ),
        "defteam",
    )
    special_teams = _rename_side(
        special.group_by([*KEYS, "posteam"]).agg(
            pl.col("epa").mean().alias("st_epa"),
            pl.col("success").mean().alias("st_success_rate"),
            pl.when(pl.col("play_type") == "field_goal")
            .then(pl.col("epa"))
            .mean()
            .alias("st_field_goal_epa"),
            pl.when(pl.col("play_type") == "extra_point")
            .then(pl.col("epa"))
            .mean()
            .alias("st_extra_point_epa"),
            pl.when(pl.col("play_type") == "punt")
            .then(pl.col("epa"))
            .mean()
            .alias("st_punt_epa"),
            pl.when(pl.col("play_type") == "kickoff")
            .then(pl.col("epa"))
            .mean()
            .alias("st_kickoff_epa"),
        ),
        "posteam",
    )

    result = base
    for frame in (
        offense_pass,
        defense_pass,
        offense_rush,
        defense_rush,
        offense_short,
        defense_short,
        offense_pressure,
        defense_pressure,
        offense_tendency,
        defense_tendency,
        special_teams,
    ):
        result = result.join(frame, on=[*KEYS, "team_id"], how="left", validate="1:1")
    return result.select(*KEYS, "team_id", *CHALLENGER_PLAY_METRICS)


def _validate_through_week(
    games: pl.DataFrame, as_of: datetime, through_week: int | None
) -> None:
    if through_week is None:
        return
    current_season = as_of.astimezone(UTC).year
    kickoff = pl.col("kickoff_utc").str.to_datetime(time_zone="UTC", strict=False)
    later_completed = games.filter(
        (pl.col("season") == current_season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") > through_week)
        & (kickoff < pl.lit(as_of.astimezone(UTC)))
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    if later_completed.height:
        raise ValueError(
            f"Completed current-season outcomes exist beyond requested through-week {through_week}"
        )


def build_challenger_features(
    as_of: datetime,
    through_week: int | None = None,
    settings: Settings | None = None,
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
            "Challenger features must use the completed synchronization retrieval time"
        )

    entries = [
        entry
        for entry in sync_manifest.get("raw_datasets", {}).values()
        if entry.get("dataset") == "pbp"
        and int(entry["season"]) >= CHALLENGER_START_SEASON
    ]
    if not entries:
        raise FileNotFoundError("Completed sync manifest has no challenger play-by-play history")

    observations: list[pl.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for entry in sorted(entries, key=lambda item: int(item["season"])):
        path = resolved.root / str(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Manifest raw file is missing: {entry['path']}")
        observations.append(_advanced_observations(pl.read_parquet(path)))
        source_hashes[f"pbp:{entry['season']}"] = str(entry["content_hash"])

    games_path = resolved.root / "games.csv"
    usage_path = resolved.root / "player_usage.csv"
    injuries_path = resolved.root / "injuries.csv"
    games = pl.read_csv(games_path, infer_schema_length=100_000)
    _validate_through_week(games, as_of_utc, through_week)
    schedule = _team_schedule(games, as_of_utc).filter(
        pl.col("season") >= CHALLENGER_START_SEASON
    )
    usage = pl.read_csv(usage_path, infer_schema_length=100_000)
    continuity = _snap_continuity(usage, games)
    observation_frame = pl.concat(observations, how="diagonal_relaxed")
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
    lags = build_lag_matrix(team_games, CHALLENGER_METRICS)
    validate_lag_boundaries(lags)
    if lags.select("game_id", "team_id").is_duplicated().any():
        raise ValueError("Duplicate challenger team-game feature keys detected")
    incomplete = lags.group_by("game_id").len().filter(pl.col("len") != 2)
    if incomplete.height:
        raise ValueError("Challenger feature store does not contain exactly two teams per game")

    injuries = pl.read_csv(injuries_path, infer_schema_length=100_000)
    lags = lags.join(
        _injury_diagnostics(injuries),
        on=["season", "week", "team_id"],
        how="left",
        validate="m:1",
    )
    feature_path = resolved.runtime_dir / "features" / "challenger_team_game_lags.parquet"
    _write_parquet_atomic(lags, feature_path)
    source_hashes.update(
        {
            "games.csv": sha256_bytes(games_path.read_bytes()),
            "player_usage.csv": sha256_bytes(usage_path.read_bytes()),
            "injuries.csv": sha256_bytes(injuries_path.read_bytes()),
        }
    )
    manifest = {
        "artifact": "data/runtime/features/challenger_team_game_lags.parquet",
        "status": "BUILT_UNWEIGHTED",
        "as_of_utc": iso_utc(as_of_utc),
        "retrieved_at_utc": iso_utc(),
        "through_week": through_week,
        "rows": lags.height,
        "seasons": sorted(int(value) for value in lags["season"].unique().to_list()),
        "development_start_season": CHALLENGER_START_SEASON,
        "metrics": list(CHALLENGER_METRICS),
        "feature_families": {
            "passing": list(PASS_METRICS),
            "rushing": list(RUSH_METRICS),
            "trenches": list(TRENCH_METRICS),
            "tendency": list(TENDENCY_METRICS),
            "special_teams": list(SPECIAL_TEAMS_METRICS),
            "continuity": list(CONTINUITY_METRICS),
        },
        "neutral_definition": "REG plays with 0.05 < pre-play wp < 0.95; kneels excluded",
        "injury_diagnostics": list(INJURY_DIAGNOSTICS),
        "injury_model_status": "DIAGNOSTIC_ONLY_UNWEIGHTED",
        "coverage_model_status": "DISABLED_EMPTY_AUTHORITATIVE_DATASET",
        "weather_model_status": "DISABLED_NO_FORECAST_VINTAGES",
        "source": "nflverse:pbp,snap-counts,injuries",
        "source_hashes": source_hashes,
        "sync_run_id": sync_manifest["run_id"],
        "schema_version": resolved.schema_version,
        "content_hash": sha256_bytes(feature_path.read_bytes()),
    }
    atomic_write_text(
        resolved.manifests_dir / "challenger_feature_inputs.latest.json",
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    return manifest
