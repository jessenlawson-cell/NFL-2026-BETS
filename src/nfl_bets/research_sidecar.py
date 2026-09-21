from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl
from nflreadpy.config import update_config

from nfl_bets.config import Settings, get_settings
from nfl_bets.util import atomic_write_text, iso_utc, sha256_bytes

SOURCE = "nflverse:nflreadpy"
CANDIDATE_STATUS = "RESEARCH_ONLY_UNWEIGHTED"
BACKFILL_STATUS = "RESEARCH_BACKFILL_NOT_OOS_ELIGIBLE"
DIAGNOSTIC_POSITIONS = ("QB", "T", "CB", "DE", "DT", "LB", "OLB")


def _require_columns(frame: pl.DataFrame, name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _numeric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _as_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _schedule_times(schedules: pl.DataFrame) -> pl.DataFrame:
    _require_columns(
        schedules,
        "schedules",
        {"game_id", "season", "week", "away_team", "home_team", "away_score", "home_score"},
    )
    if schedules.select("game_id").is_duplicated().any():
        raise ValueError("Duplicate schedule game_id values")
    if "kickoff_utc" in schedules.columns:
        kickoff = pl.col("kickoff_utc").str.to_datetime(format="%+", time_zone="UTC", strict=True)
    else:
        _require_columns(schedules, "schedules", {"gameday", "gametime"})
        kickoff = (
            pl.concat_str([pl.col("gameday"), pl.lit("T"), pl.col("gametime")])
            .str.to_datetime(
                format="%Y-%m-%dT%H:%M", time_zone="America/New_York", strict=True
            )
            .dt.convert_time_zone("UTC")
        )
    return schedules.with_columns(kickoff.alias("_kickoff_utc"))


def _unique_crosswalk(players: pl.DataFrame) -> pl.DataFrame:
    _require_columns(players, "players", {"gsis_id", "pfr_id"})
    crosswalk = players.select("gsis_id", "pfr_id").filter(
        pl.col("gsis_id").is_not_null() & pl.col("pfr_id").is_not_null()
    )
    if (
        crosswalk.select("gsis_id").is_duplicated().any()
        or crosswalk.select("pfr_id").is_duplicated().any()
    ):
        raise ValueError("Duplicate GSIS or PFR identifiers in player crosswalk")
    return crosswalk


def _qb_pregame_rate(
    pbp: pl.DataFrame,
    schedule_times: pl.DataFrame,
    player_gsis_id: str,
    decision_time: datetime,
    half_life: float,
    prior_strength: float,
) -> float | None:
    _require_columns(pbp, "pbp", {"game_id", "passer_player_id", "qb_dropback", "epa"})
    prior = (
        pbp.join(
            schedule_times.select("game_id", "_kickoff_utc"),
            on="game_id",
            how="inner",
            validate="m:1",
        )
        .filter(
            (pl.col("qb_dropback") == 1)
            & pl.col("epa").is_not_null()
            & (pl.col("_kickoff_utc") < decision_time)
        )
    )
    player = prior.filter(pl.col("passer_player_id") == player_gsis_id)
    if player.is_empty():
        return None
    games = (
        player.group_by("game_id", "_kickoff_utc")
        .agg(pl.col("epa").sum().alias("epa_sum"), pl.len().alias("dropbacks"))
        .sort("_kickoff_utc", descending=True)
        .with_row_index("games_ago")
        .with_columns(
            pl.lit(2.0)
            .pow(-pl.col("games_ago").cast(pl.Float64) / half_life)
            .alias("weight")
        )
    )
    totals = games.select(
        (pl.col("epa_sum") * pl.col("weight")).sum().alias("weighted_epa"),
        (pl.col("dropbacks") * pl.col("weight")).sum().alias("weighted_dropbacks"),
    ).row(0, named=True)
    prior_mean = prior["epa"].mean()
    if prior_mean is None:
        return None
    numerator = _numeric(totals["weighted_epa"], "weighted_epa") + prior_strength * _numeric(
        prior_mean, "prior_mean"
    )
    denominator = _numeric(totals["weighted_dropbacks"], "weighted_dropbacks") + prior_strength
    return numerator / denominator


def build_qb_net_leverage(
    *,
    schedules: pl.DataFrame,
    injuries: pl.DataFrame,
    depth_charts: pl.DataFrame,
    players: pl.DataFrame,
    snap_counts: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    team_id: str,
    decision_time: datetime,
    source_available_at: datetime,
    half_life: float,
    prior_strength: float,
) -> dict[str, object]:
    """Build one leakage-safe, research-only QB replacement shock observation."""
    cutoff = _as_utc(decision_time, "decision_time")
    available = _as_utc(source_available_at, "source_available_at")
    if available > cutoff:
        raise ValueError("source_available_at must not exceed decision_time")
    if half_life <= 0 or prior_strength < 0:
        raise ValueError("half_life must be positive and prior_strength non-negative")
    base: dict[str, object] = {
        "season": season,
        "week": week,
        "team_id": team_id,
        "decision_time_utc": iso_utc(cutoff),
        "source_available_at_utc": iso_utc(available),
        "evidence_status": "PROSPECTIVE_ELIGIBLE",
        "candidate_status": CANDIDATE_STATUS,
        "decision": "PASS",
        "starter_player_gsis_id": None,
        "backup_player_gsis_id": None,
        "lost_snap_share_4g": None,
        "starter_pregame_epa_per_dropback": None,
        "backup_pregame_epa_per_dropback": None,
        "qb_net_le_shock": None,
    }

    _require_columns(
        depth_charts, "depth_charts", {"dt", "team", "gsis_id", "pos_abb", "pos_rank"}
    )
    depth = depth_charts.with_columns(
        pl.col("dt").str.to_datetime(format="%+", time_zone="UTC", strict=True).alias("_dt")
    ).filter(
        (pl.col("team") == team_id)
        & (pl.col("pos_abb") == "QB")
        & (pl.col("_dt") <= cutoff)
    )
    if depth.is_empty():
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_DEPTH"}
    latest = depth.filter(pl.col("_dt") == depth["_dt"].max())
    starter_rows = latest.filter(pl.col("pos_rank") == 1)
    backup_rows = latest.filter(pl.col("pos_rank") == 2)
    if starter_rows.height != 1 or backup_rows.height != 1:
        return {**base, "eligibility_status": "INELIGIBLE_AMBIGUOUS_DEPTH"}
    starter_id = starter_rows["gsis_id"][0]
    backup_id = backup_rows["gsis_id"][0]
    if starter_id is None or backup_id is None:
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_DEPTH_ID"}
    base.update(
        starter_player_gsis_id=str(starter_id), backup_player_gsis_id=str(backup_id)
    )

    _require_columns(
        injuries,
        "injuries",
        {"season", "week", "team", "gsis_id", "position", "report_status"},
    )
    team_week_injuries = injuries.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("team") == team_id)
    )
    if team_week_injuries.is_empty():
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_INJURY_COVERAGE"}
    qualifying = team_week_injuries.filter(
        (pl.col("gsis_id") == starter_id)
        & (pl.col("position") == "QB")
        & (pl.col("report_status") == "Out")
    )
    if qualifying.is_empty():
        return {
            **base,
            "eligibility_status": "NO_QUALIFYING_EVENT",
            "qb_net_le_shock": 0.0,
        }
    if qualifying.height != 1:
        return {**base, "eligibility_status": "INELIGIBLE_AMBIGUOUS_INJURY"}

    crosswalk = _unique_crosswalk(players)
    starter_map = crosswalk.filter(pl.col("gsis_id") == starter_id)
    if starter_map.height != 1:
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_STARTER_CROSSWALK"}
    schedule_times = _schedule_times(schedules)
    completed = schedule_times.filter(
        ((pl.col("away_team") == team_id) | (pl.col("home_team") == team_id))
        & pl.col("away_score").is_not_null()
        & pl.col("home_score").is_not_null()
        & (pl.col("_kickoff_utc") < cutoff)
    ).sort("_kickoff_utc", descending=True).head(4)
    if completed.height != 4:
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_STARTER_VOLUME"}

    _require_columns(
        snap_counts, "snap_counts", {"game_id", "pfr_player_id", "team", "offense_pct"}
    )
    starter_pfr_id = starter_map["pfr_id"][0]
    starter_snaps = snap_counts.filter(
        (pl.col("pfr_player_id") == starter_pfr_id) & (pl.col("team") == team_id)
    ).join(
        completed.select("game_id", "_kickoff_utc"),
        on="game_id",
        how="inner",
        validate="1:1",
    )
    if starter_snaps.height != 4 or starter_snaps["offense_pct"].null_count():
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_STARTER_VOLUME"}
    lost_volume = starter_snaps["offense_pct"].mean()
    lost_volume_value = _numeric(lost_volume, "lost_snap_share_4g")
    base["lost_snap_share_4g"] = lost_volume_value

    starter_rate = _qb_pregame_rate(
        pbp, schedule_times, str(starter_id), cutoff, half_life, prior_strength
    )
    backup_rate = _qb_pregame_rate(
        pbp, schedule_times, str(backup_id), cutoff, half_life, prior_strength
    )
    base["starter_pregame_epa_per_dropback"] = starter_rate
    base["backup_pregame_epa_per_dropback"] = backup_rate
    if starter_rate is None:
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_STARTER_RATE"}
    if backup_rate is None:
        return {**base, "eligibility_status": "INELIGIBLE_MISSING_BACKUP_RATE"}
    base["qb_net_le_shock"] = lost_volume_value * (backup_rate - starter_rate)
    return {**base, "eligibility_status": "ELIGIBLE_RESEARCH_ONLY"}


def game_qb_net_leverage_diff(
    home: Mapping[str, object], away: Mapping[str, object]
) -> float | None:
    home_value = home.get("qb_net_le_shock")
    away_value = away.get("qb_net_le_shock")
    if home_value is None or away_value is None:
        return None
    return _numeric(home_value, "home qb_net_le_shock") - _numeric(
        away_value, "away qb_net_le_shock"
    )


def standardize_ablation_fold(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    column: str = "qb_net_le_shock",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Standardize one ablation feature using train-fold statistics only."""
    _require_columns(train, "train", {column})
    _require_columns(validation, "validation", {column})
    stats = train.select(
        pl.col(column).mean().alias("mean"),
        ((pl.col(column) - pl.col(column).mean()).pow(2).mean().sqrt()).alias("std"),
    ).row(0, named=True)
    mean = stats["mean"]
    std = stats["std"]
    if mean is None or std is None or float(std) == 0.0:
        raise ValueError("Training fold must contain non-constant non-null feature values")
    scaled = f"{column}_z"
    expression = ((pl.col(column) - float(mean)) / float(std)).alias(scaled)
    return train.with_columns(expression), validation.with_columns(expression)


def _probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    row_brier = (probability - actual) ** 2
    row_log_loss = -(
        actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability)
    )
    bins = np.minimum((probability * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_id in range(10):
        members = bins == bin_id
        if members.any():
            calibration_error += float(members.mean()) * abs(
                float(probability[members].mean()) - float(actual[members].mean())
            )
    return {
        "brier": float(row_brier.mean()),
        "log_loss": float(row_log_loss.mean()),
        "calibration_error": calibration_error,
    }


def evaluate_matched_ablation(
    observations: pl.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 202603,
) -> dict[str, object]:
    """Evaluate precomputed matched baseline/candidate probabilities without wagering output."""
    required = {
        "game_id",
        "actual",
        "baseline_probability",
        "candidate_probability",
        "qb_net_le_shock",
        "evidence_status",
    }
    _require_columns(observations, "observations", required)
    if observations.is_empty() or observations.select(list(required)).null_count().sum_horizontal()[
        0
    ]:
        raise ValueError("Ablation evaluation requires matched non-null rows")
    if observations.select("game_id").is_duplicated().any():
        raise ValueError("Ablation evaluation requires unique game_id rows")
    if not observations["evidence_status"].eq("PROSPECTIVE_ELIGIBLE").all():
        raise ValueError("Ablation evaluation requires prospectively eligible evidence")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    actual = observations["actual"].cast(pl.Float64).to_numpy()
    baseline = observations["baseline_probability"].cast(pl.Float64).to_numpy()
    candidate = observations["candidate_probability"].cast(pl.Float64).to_numpy()
    if not np.isin(actual, [0.0, 1.0]).all():
        raise ValueError("actual must contain only binary outcomes")
    if not (
        np.isfinite(baseline).all()
        and np.isfinite(candidate).all()
        and ((baseline > 0.0) & (baseline < 1.0)).all()
        and ((candidate > 0.0) & (candidate < 1.0)).all()
    ):
        raise ValueError("Probabilities must be finite and strictly between zero and one")

    baseline_metrics = _probability_metrics(actual, baseline)
    candidate_metrics = _probability_metrics(actual, candidate)
    delta = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in ("brier", "log_loss", "calibration_error")
    }
    brier_rows = (candidate - actual) ** 2 - (baseline - actual) ** 2
    log_rows = -(
        actual * np.log(candidate) + (1.0 - actual) * np.log(1.0 - candidate)
    ) + (actual * np.log(baseline) + (1.0 - actual) * np.log(1.0 - baseline))
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, observations.height, size=(bootstrap_samples, observations.height)
    )
    brier_interval = np.quantile(brier_rows[indices].mean(axis=1), [0.025, 0.975])
    log_interval = np.quantile(log_rows[indices].mean(axis=1), [0.025, 0.975])
    acceptance = bool(
        delta["brier"] < 0.0
        and delta["log_loss"] < 0.0
        and brier_interval[1] < 0.0
        and log_interval[1] < 0.0
        and delta["calibration_error"] <= 0.0
    )
    return {
        "status": CANDIDATE_STATUS,
        "decision": "PASS",
        "rows": observations.height,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "paired_bootstrap_95pct": {
            "brier_delta": [float(value) for value in brier_interval],
            "log_loss_delta": [float(value) for value in log_interval],
        },
        "meets_research_acceptance": acceptance,
    }


def _capture_id(captured_at: datetime) -> str:
    return captured_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _schema_hash(frame: pl.DataFrame) -> str:
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    return sha256_bytes(json.dumps(schema, separators=(",", ":")).encode("utf-8"))


def _write_parquet_atomic(frame: pl.DataFrame, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    try:
        frame.write_parquet(temporary_name, compression="zstd")
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return sha256_bytes(target.read_bytes())


def _protected_files(settings: Settings) -> Iterable[Path]:
    for pattern in ("*.csv", "MODEL_SPEC*.md", "CHALLENGER_SPEC.md"):
        yield from sorted(settings.root.glob(pattern))
    for root in (
        settings.artifacts_dir / "models",
        settings.runtime_dir,
        settings.manifests_dir,
        settings.reports_dir,
    ):
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "research" not in path.relative_to(root).parts:
                yield path


def _database_counts(settings: Settings) -> dict[str, int]:
    if not settings.db_path.exists():
        return {}
    uri = f"{settings.db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {
            str(name): int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for (name,) in tables
        }


def _operational_state(settings: Settings) -> dict[str, Any]:
    files = {
        path.resolve().relative_to(settings.root).as_posix(): sha256_bytes(path.read_bytes())
        for path in _protected_files(settings)
    }
    return {"files": files, "database_row_counts": _database_counts(settings)}


def _load_frames(seasons: list[int]) -> dict[str, pl.DataFrame]:
    return {
        "schedules": nfl.load_schedules(seasons),
        "pbp": nfl.load_pbp(seasons),
        "injuries": nfl.load_injuries(seasons),
        "depth_charts": nfl.load_depth_charts(seasons),
        "players": nfl.load_players(),
        "snap_counts": nfl.load_snap_counts(seasons),
        "ngs_passing": nfl.load_nextgen_stats(seasons, stat_type="passing"),
        "ngs_rushing": nfl.load_nextgen_stats(seasons, stat_type="rushing"),
        "ngs_receiving": nfl.load_nextgen_stats(seasons, stat_type="receiving"),
        "ftn_charting": nfl.load_ftn_charting(seasons),
    }


def _position_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(row[column]): int(row["rows"])
        for row in (
            frame.filter(pl.col(column).is_in(DIAGNOSTIC_POSITIONS))
            .group_by(column)
            .agg(pl.len().alias("rows"))
            .sort(column)
            .to_dicts()
        )
    }


def _personnel_coverage(frames: Mapping[str, pl.DataFrame]) -> dict[str, object]:
    injuries = frames["injuries"]
    depth = frames["depth_charts"]
    snaps = frames["snap_counts"]
    players = frames["players"]
    coverage: dict[str, object] = {
        "status": "COVERAGE_ONLY_UNWEIGHTED",
        "pass_rusher_role_status": "UNDEFINED_RESEARCH_DIAGNOSTIC",
        "injury_rows_by_position": _position_counts(injuries, "position"),
        "depth_rows_by_position": _position_counts(depth, "pos_abb"),
        "snap_rows_by_position": _position_counts(snaps, "position"),
        "mapped_snap_rows_by_position": {},
        "unmapped_snap_rows_by_position": {},
        "crosswalk_status": "UNAVAILABLE_REQUIRED_COLUMNS",
    }
    if not {"pfr_player_id", "position"}.issubset(snaps.columns) or not {
        "pfr_id",
        "gsis_id",
    }.issubset(players.columns):
        return coverage
    crosswalk = players.select("pfr_id", "gsis_id").filter(
        pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null()
    )
    if (
        crosswalk.select("pfr_id").is_duplicated().any()
        or crosswalk.select("gsis_id").is_duplicated().any()
    ):
        coverage["crosswalk_status"] = "DUPLICATE_IDENTIFIERS"
        return coverage
    joined = snaps.join(
        crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left", validate="m:1"
    ).filter(pl.col("position").is_in(DIAGNOSTIC_POSITIONS))
    coverage["crosswalk_status"] = "VALID_UNIQUE_IDS"
    coverage["mapped_snap_rows_by_position"] = _position_counts(
        joined.filter(pl.col("gsis_id").is_not_null()), "position"
    )
    coverage["unmapped_snap_rows_by_position"] = _position_counts(
        joined.filter(pl.col("gsis_id").is_null()), "position"
    )
    return coverage


def capture_research_snapshot(
    seasons: list[int],
    *,
    settings: Settings | None = None,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Capture immutable challenger-0.3 research inputs without operational writes."""
    if not seasons or any(not isinstance(season, int) for season in seasons):
        raise ValueError("At least one integer season is required")
    instant = captured_at or datetime.now(UTC)
    if instant.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    instant = instant.astimezone(UTC)
    resolved = settings or get_settings()
    capture_id = _capture_id(instant)
    research_root = resolved.data_dir / "research" / "challenger_0_3"
    stage_root = research_root / f".staging-{capture_id}"
    final_root = research_root / "raw" / capture_id
    if stage_root.exists() or final_root.exists():
        raise FileExistsError(f"Research capture {capture_id} already exists")

    before = _operational_state(resolved)
    update_config(
        cache_mode="filesystem",
        cache_dir=resolved.cache_dir,
        cache_duration=86_400,
        verbose=False,
        timeout=60,
    )
    stage_root.mkdir(parents=True, exist_ok=False)
    try:
        frames = _load_frames(sorted(set(seasons)))
        datasets: dict[str, dict[str, Any]] = {}
        total_rows = 0
        for name, frame in frames.items():
            target = stage_root / f"{name}.parquet"
            content_hash = _write_parquet_atomic(frame, target)
            total_rows += frame.height
            datasets[name] = {
                "source": SOURCE,
                "rows": frame.height,
                "columns": frame.columns,
                "schema_hash": _schema_hash(frame),
                "content_hash": content_hash,
                "retrieved_at_utc": iso_utc(instant),
                "source_available_at_utc": iso_utc(instant),
                "historical_availability_status": BACKFILL_STATUS,
                "path": (
                    final_root / f"{name}.parquet"
                ).resolve().relative_to(resolved.root).as_posix(),
            }

        after = _operational_state(resolved)
        if after != before:
            raise RuntimeError("Research capture changed protected operational state")

        personnel_coverage = _personnel_coverage(frames)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_root, final_root)
        manifest = {
            "capture_id": capture_id,
            "candidate_version": "challenger-0.3.0",
            "candidate_status": CANDIDATE_STATUS,
            "decision": "PASS",
            "captured_at_utc": iso_utc(instant),
            "seasons": sorted(set(seasons)),
            "operational_state_unchanged": True,
            "protected_state": after,
            "datasets": datasets,
            "personnel_coverage": personnel_coverage,
        }
        manifest_path = (
            resolved.manifests_dir / "research" / "challenger_0_3" / f"{capture_id}.json"
        )
        report_path = (
            resolved.reports_dir / "research" / "challenger_0_3" / f"{capture_id}.json"
        )
        atomic_write_text(manifest_path, json.dumps(manifest, sort_keys=True, indent=2))
        atomic_write_text(
            report_path,
            json.dumps(
                {
                    "capture_id": capture_id,
                    "status": CANDIDATE_STATUS,
                    "decision": "PASS",
                    "datasets": len(datasets),
                    "rows": total_rows,
                    "manifest": manifest_path.relative_to(resolved.root).as_posix(),
                    "personnel_coverage": personnel_coverage,
                },
                sort_keys=True,
                indent=2,
            ),
        )
        return {
            "capture_id": capture_id,
            "status": CANDIDATE_STATUS,
            "decision": "PASS",
            "datasets": len(datasets),
            "rows": total_rows,
            "manifest": manifest_path.relative_to(resolved.root).as_posix(),
            "report": report_path.relative_to(resolved.root).as_posix(),
        }
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _main() -> None:
    parser = argparse.ArgumentParser(description="Capture challenger-0.3 research inputs")
    parser.add_argument("--season", action="append", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(capture_research_snapshot(args.season), sort_keys=True))


if __name__ == "__main__":
    _main()
