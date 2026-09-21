from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import initialize_database, transaction
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.teams import canonical_team
from nfl_bets.util import atomic_write_text, canonical_hash, iso_utc, sha256_bytes

SOURCE = "nflverse:nflreadpy"


def _metadata(
    record: dict[str, object], retrieved_at: str, schema_version: str
) -> dict[str, object]:
    record.update(
        source=SOURCE,
        retrieved_at_utc=retrieved_at,
        source_updated_at_utc=None,
        schema_version=schema_version,
    )
    record["content_hash"] = canonical_hash(record)
    return record


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


def _schedule_records(
    frame: pl.DataFrame, retrieved_at: str, schema_version: str
) -> list[dict[str, object]]:
    regular = frame.filter(pl.col("game_type") == "REG")
    records: list[dict[str, object]] = []
    for row in regular.iter_rows(named=True):
        gametime = row.get("gametime") or "00:00"
        raw_kickoff = f"{row['gameday']}T{gametime}:00"
        try:
            kickoff = datetime.fromisoformat(raw_kickoff).replace(
                tzinfo=ZoneInfo("America/New_York")
            )
            kickoff_utc = iso_utc(kickoff)
        except ValueError:
            kickoff_utc = raw_kickoff
        record = {
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "game_type": row["game_type"],
            "kickoff_utc": kickoff_utc,
            "away_team": canonical_team(row["away_team"]),
            "home_team": canonical_team(row["home_team"]),
            "away_score": row.get("away_score"),
            "home_score": row.get("home_score"),
            "result": row.get("result"),
            "total": row.get("total"),
            "away_rest": row.get("away_rest"),
            "home_rest": row.get("home_rest"),
            "away_moneyline": row.get("away_moneyline"),
            "home_moneyline": row.get("home_moneyline"),
            "spread_line": row.get("spread_line"),
            "away_spread_odds": row.get("away_spread_odds"),
            "home_spread_odds": row.get("home_spread_odds"),
            "total_line": row.get("total_line"),
            "under_odds": row.get("under_odds"),
            "over_odds": row.get("over_odds"),
            "roof": row.get("roof"),
            "surface": row.get("surface"),
            "temp": row.get("temp"),
            "wind": row.get("wind"),
        }
        records.append(_metadata(record, retrieved_at, schema_version))
    return records


def _injury_records(
    frame: pl.DataFrame, retrieved_at: str, schema_version: str
) -> list[dict[str, object]]:
    if frame.is_empty():
        return []
    regular = frame.filter(pl.col("season_type") == "REG")
    records: list[dict[str, object]] = []
    for row in regular.iter_rows(named=True):
        player_id = row.get("gsis_id") or f"UNRESOLVED:{row.get('full_name', '')}"
        record = {
            "season": row["season"],
            "week": row["week"],
            "team_id": canonical_team(row["team"]),
            "player_id": player_id,
            "full_name": row.get("full_name"),
            "position": row.get("position"),
            "injury_status": row.get("report_status"),
            "practice_status": row.get("practice_status"),
            "report_primary_injury": row.get("report_primary_injury"),
            "report_secondary_injury": row.get("report_secondary_injury"),
            "snap_share_impact": None,
            "replacement_quality": None,
        }
        records.append(_metadata(record, retrieved_at, schema_version))
    return records


def _usage_records(
    frame: pl.DataFrame, retrieved_at: str, schema_version: str
) -> list[dict[str, object]]:
    if frame.is_empty():
        return []
    regular = frame.filter(pl.col("game_type") == "REG")
    records: list[dict[str, object]] = []
    for row in regular.iter_rows(named=True):
        player_id = row.get("pfr_player_id") or f"UNRESOLVED:{row.get('player', '')}"
        record = {
            "season": row["season"],
            "week": row["week"],
            "game_id": row["game_id"],
            "player_id": player_id,
            "player_name": row.get("player"),
            "team_id": canonical_team(row["team"]),
            "position": row.get("position"),
            "offense_snaps": row.get("offense_snaps"),
            "offense_pct": row.get("offense_pct"),
            "defense_snaps": row.get("defense_snaps"),
            "defense_pct": row.get("defense_pct"),
            "special_teams_snaps": row.get("st_snaps"),
            "special_teams_pct": row.get("st_pct"),
        }
        records.append(_metadata(record, retrieved_at, schema_version))
    return records


def _records_frame(name: str, records: list[dict[str, object]]) -> pl.DataFrame:
    columns = ARTIFACT_SCHEMAS[name].columns
    if not records:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(records).select(columns)


def _deduplicate(name: str, records: list[dict[str, object]]) -> list[dict[str, object]]:
    primary_key = ARTIFACT_SCHEMAS[name].primary_key
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        unique[tuple(record[column] for column in primary_key)] = record
    return sorted(unique.values(), key=lambda row: tuple(str(row[key]) for key in primary_key))


def _logical_hash(records: list[dict[str, object]]) -> str:
    hashes = sorted(str(record["content_hash"]) for record in records)
    return sha256_bytes("\n".join(hashes).encode("ascii"))


def _validate_staged(name: str, frame: pl.DataFrame) -> None:
    schema = ARTIFACT_SCHEMAS[name]
    if tuple(frame.columns) != schema.columns:
        raise ValueError(f"Staged {name}.csv does not match schema {schema.name}")
    if frame.height:
        duplicates = frame.group_by(schema.primary_key).len().filter(pl.col("len") > 1)
        if not duplicates.is_empty():
            raise ValueError(f"Staged {name}.csv contains duplicate primary keys")
        for column in ("source", "retrieved_at_utc", "schema_version", "content_hash"):
            if frame[column].null_count():
                raise ValueError(f"Staged {name}.csv is missing required {column} values")


def _replace_table(
    connection: sqlite3.Connection, table: str, records: list[dict[str, object]]
) -> None:
    connection.execute(f"DELETE FROM {table}")
    if not records:
        return
    columns = ARTIFACT_SCHEMAS[table].columns
    placeholders = ",".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        [tuple(record.get(column) for column in columns) for record in records],
    )


def _ensure_empty_artifacts(settings: Settings) -> None:
    for name in ARTIFACT_SCHEMAS:
        path = settings.root / f"{name}.csv"
        if not path.exists():
            atomic_write_text(path, _records_frame(name, []).write_csv())


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_manifest(settings: Settings, run_id: str, payload: dict[str, Any]) -> Path:
    run_path = settings.manifests_dir / f"nflverse_sync_{run_id}.json"
    text = json.dumps(payload, sort_keys=True, indent=2)
    atomic_write_text(run_path, text)
    if payload.get("status") == "COMPLETE":
        atomic_write_text(settings.manifests_dir / "nflverse_sync.latest.json", text)
    return run_path


def sync_data(
    through: int,
    start_season: int = 2009,
    include_pbp: bool = True,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Download season slices, validate staged outputs, and promote canonical artifacts."""
    resolved = settings or get_settings()
    resolved.ensure_directories()
    if through < start_season:
        raise ValueError("--through must be greater than or equal to --start-season")
    if through > datetime.now(resolved.tz).year:
        raise ValueError("--through is implausibly far in the future")
    if through < 2012:
        raise ValueError("--through must be at least 2012")

    initialize_database(resolved)
    update_config(
        cache_mode="filesystem",
        cache_dir=resolved.cache_dir,
        cache_duration=86_400,
        verbose=False,
        timeout=60,
    )
    seasons = list(range(start_season, through + 1))
    retrieved_at = iso_utc()
    run_id = str(uuid.uuid4())
    stage_root = resolved.staging_dir / run_id
    stage_raw = stage_root / "raw" / "nflverse"
    stage_authoritative = stage_root / "authoritative"
    stage_raw.mkdir(parents=True, exist_ok=False)
    stage_authoritative.mkdir(parents=True, exist_ok=False)
    with transaction(resolved) as connection:
        connection.execute(
            "INSERT INTO ingestion_runs(run_id,dataset,started_at_utc,status,seasons_json,source) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, "nflverse-core", retrieved_at, "STARTED", json.dumps(seasons), SOURCE),
        )

    totals: dict[str, int] = {
        "schedules": 0,
        "team_stats": 0,
        "injuries": 0,
        "snap_counts": 0,
    }
    if include_pbp:
        totals["pbp"] = 0
    availability: dict[str, list[dict[str, Any]]] = {name: [] for name in totals}
    frames: dict[str, list[pl.DataFrame]] = {name: [] for name in totals}
    raw_entries: dict[str, dict[str, Any]] = {}

    def loaders_for(season: int) -> dict[str, tuple[Callable[[], pl.DataFrame], bool]]:
        loaders: dict[str, tuple[Callable[[], pl.DataFrame], bool]] = {
            "schedules": (lambda: nfl.load_schedules([season]), True),
            "team_stats": (
                lambda: nfl.load_team_stats([season], summary_level="week"),
                False,
            ),
            "injuries": (lambda: nfl.load_injuries([season]), False),
        }
        if season >= 2012:
            loaders["snap_counts"] = (lambda: nfl.load_snap_counts([season]), False)
        if include_pbp:
            loaders["pbp"] = (lambda: nfl.load_pbp([season]), True)
        return loaders

    try:
        for season in seasons:
            if season < 2012:
                availability["snap_counts"].append(
                    {
                        "season": season,
                        "status": "NOT_AVAILABLE",
                        "reason": "nflverse snap counts begin in 2012",
                    }
                )
            for name, (loader, required) in loaders_for(season).items():
                try:
                    frame = loader()
                except Exception as exc:
                    availability[name].append(
                        {"season": season, "status": "ERROR", "reason": str(exc)[:500]}
                    )
                    if required:
                        raise RuntimeError(f"Required {name} failed for {season}") from exc
                    continue
                if frame.is_empty():
                    availability[name].append(
                        {"season": season, "status": "EMPTY", "reason": "source returned no rows"}
                    )
                    if required:
                        raise RuntimeError(f"Required {name} returned no rows for {season}")
                    continue
                raw_path = stage_raw / f"{name}_{season}.parquet"
                raw_hash = _write_parquet_atomic(frame, raw_path)
                frames[name].append(frame)
                totals[name] += frame.height
                availability[name].append(
                    {"season": season, "status": "AVAILABLE", "rows": frame.height}
                )
                raw_entries[f"{name}:{season}"] = {
                    "dataset": name,
                    "season": season,
                    "rows": frame.height,
                    "content_hash": raw_hash,
                    "source_updated_at_utc": None,
                    "source_update_status": "UNAVAILABLE_FROM_PROVIDER_LIBRARY",
                    "path": f"data/raw/nflverse/{name}_{season}.parquet",
                }

        schedules = pl.concat(frames["schedules"], how="diagonal_relaxed")
        injury_frame = (
            pl.concat(frames["injuries"], how="diagonal_relaxed")
            if frames["injuries"]
            else pl.DataFrame()
        )
        snap_frame = (
            pl.concat(frames["snap_counts"], how="diagonal_relaxed")
            if frames["snap_counts"]
            else pl.DataFrame()
        )
        curated_records = {
            "games": _deduplicate(
                "games", _schedule_records(schedules, retrieved_at, resolved.schema_version)
            ),
            "injuries": _deduplicate(
                "injuries", _injury_records(injury_frame, retrieved_at, resolved.schema_version)
            ),
            "player_usage": _deduplicate(
                "player_usage", _usage_records(snap_frame, retrieved_at, resolved.schema_version)
            ),
        }
        curated_frames: dict[str, pl.DataFrame] = {}
        curated_manifest: dict[str, dict[str, Any]] = {}
        for name, records in curated_records.items():
            frame = _records_frame(name, records)
            _validate_staged(name, frame)
            atomic_write_text(stage_authoritative / f"{name}.csv", frame.write_csv())
            curated_frames[name] = frame
            curated_manifest[name] = {
                "rows": frame.height,
                "content_hash": _logical_hash(records),
                "path": f"{name}.csv",
            }

        for staged_path in sorted(stage_raw.glob("*.parquet")):
            target = resolved.raw_dir / "nflverse" / staged_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, target)

        for name, frame in curated_frames.items():
            atomic_write_text(resolved.root / f"{name}.csv", frame.write_csv())
        empty_metrics = _records_frame("team_metrics", [])
        atomic_write_text(resolved.root / "team_metrics.csv", empty_metrics.write_csv())

        with transaction(resolved) as connection:
            connection.execute("DELETE FROM team_metrics")
            for name in ("player_usage", "injuries", "games"):
                _replace_table(connection, name, curated_records[name])
            connection.execute(
                "UPDATE ingestion_runs SET completed_at_utc=?,status='COMPLETE',row_count=? "
                "WHERE run_id=?",
                (iso_utc(), sum(len(rows) for rows in curated_records.values()), run_id),
            )

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "status": "COMPLETE",
            "source": SOURCE,
            "retrieved_at_utc": retrieved_at,
            "source_updated_at_utc": None,
            "source_update_status": "UNAVAILABLE_FROM_PROVIDER_LIBRARY",
            "schema_version": resolved.schema_version,
            "seasons": seasons,
            "include_pbp": include_pbp,
            "availability": availability,
            "raw_datasets": raw_entries,
            "authoritative_artifacts": curated_manifest,
        }
        _write_manifest(resolved, run_id, manifest)
        shutil.rmtree(stage_root)
        _ensure_empty_artifacts(resolved)
        return totals
    except BaseException as exc:
        completed_at = iso_utc()
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE ingestion_runs SET completed_at_utc=?,status='FAILED',error_message=? "
                "WHERE run_id=?",
                (completed_at, str(exc)[:2000], run_id),
            )
        _write_manifest(
            resolved,
            run_id,
            {
                "run_id": run_id,
                "status": "FAILED",
                "source": SOURCE,
                "retrieved_at_utc": retrieved_at,
                "completed_at_utc": completed_at,
                "schema_version": resolved.schema_version,
                "seasons": seasons,
                "include_pbp": include_pbp,
                "availability": availability,
                "error": str(exc)[:2000],
                "staging_path": _relative(stage_root, resolved.root),
            },
        )
        raise
