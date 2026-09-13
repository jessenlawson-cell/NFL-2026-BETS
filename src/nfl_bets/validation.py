from __future__ import annotations

from typing import Any

import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database, table_columns
from nfl_bets.schemas import ARTIFACT_SCHEMAS


class DataQualityError(RuntimeError):
    pass


def validate_all(settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
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
                raise DataQualityError(f"{name}.csv columns do not match schema version 1.0.0")
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
            results[name] = {"rows": frame.height, "schema": "VALID", "duplicates": 0}

    games = loaded["games"]
    metrics = loaded["team_metrics"]
    if metrics.height:
        unmatched = metrics.join(games.select("game_id"), on="game_id", how="anti")
        if not unmatched.is_empty():
            raise DataQualityError("team_metrics.csv contains unmatched game_id values")
        kickoff_check = metrics.join(
            games.select("game_id", pl.col("kickoff_utc").alias("game_kickoff_utc")),
            on="game_id",
            how="left",
            validate="m:1",
        ).filter(pl.col("kickoff_utc") != pl.col("game_kickoff_utc"))
        if not kickoff_check.is_empty():
            raise DataQualityError("team_metrics kickoff does not match canonical games kickoff")
    results["status"] = "VALID"
    return results
