from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_bets.config import Settings
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.validation import validate_all


def audit_repository(root: Path, as_of: datetime) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise ValueError("analysis as-of timestamp must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    frames: dict[str, pd.DataFrame] = {}
    datasets: dict[str, Any] = {}
    unmatched: dict[str, int] = {}

    for name, schema in ARTIFACT_SCHEMAS.items():
        path = root / f"{name}.csv"
        if not path.exists():
            raise RuntimeError(f"DATA QUALITY FAILURE: missing {path.name}")
        frame = pd.read_csv(path, low_memory=False)
        if tuple(frame.columns) != schema.columns:
            raise RuntimeError(f"DATA QUALITY FAILURE: {path.name} schema mismatch")
        duplicate_count = int(frame.duplicated(list(schema.primary_key)).sum())
        metadata_missing = {
            column: int(frame[column].isna().sum())
            for column in ("source", "retrieved_at_utc", "schema_version", "content_hash")
        }
        if duplicate_count:
            raise RuntimeError(f"DATA QUALITY FAILURE: {path.name} duplicate primary keys")
        if any(metadata_missing.values()):
            raise RuntimeError(f"DATA QUALITY FAILURE: {path.name} missing required metadata")
        frames[name] = frame
        datasets[name] = {
            "shape": list(frame.shape),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "missingness": {
                column: int(value) for column, value in frame.isna().sum().items() if int(value)
            },
            "duplicate_primary_keys": duplicate_count,
        }

    games = frames["games"]
    odds = frames["market_odds"]
    predictions = frames["model_predictions"]
    for name in ("team_metrics", "market_odds", "player_usage", "model_predictions"):
        frame = frames[name]
        if frame.empty:
            unmatched[name] = 0
            continue
        joined = frame[["game_id"]].merge(
            games[["game_id"]], on="game_id", how="left", validate="many_to_one", indicator=True
        )
        unmatched[name] = int(joined["_merge"].eq("left_only").sum())
        if unmatched[name]:
            raise RuntimeError(f"DATA QUALITY FAILURE: {name}.csv has unmatched game_id rows")

    state = (root / "PROJECT_STATE.md").read_text(encoding="utf-8")
    fields = {
        key: re.search(rf"\*\*{label}:\*\*\s*([^\r\n]+)", state)
        for key, label in (
            ("season", "Current Season"),
            ("week", "Current Week"),
            ("model", "Last Model Version"),
            ("betting", "Betting Status"),
        )
    }
    if any(match is None for match in fields.values()):
        raise RuntimeError("DATA QUALITY FAILURE: PROJECT_STATE.md is incomplete")
    season = int(fields["season"].group(1))  # type: ignore[union-attr]
    week = int(fields["week"].group(1))  # type: ignore[union-attr]
    model_version = fields["model"].group(1).split()[0]  # type: ignore[union-attr]
    betting_status = fields["betting"].group(1)  # type: ignore[union-attr]

    reasons: list[str] = []
    if betting_status.lower().startswith("pass-only"):
        reasons.append("PROJECT_STATE betting status is PASS-only")
    policy_path = root / "manifests" / f"prospective_policy_{model_version}.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("allowed_decisions") == ["PASS"]:
        reasons.append("prospective policy permits only PASS decisions")

    latest_data = max(
        (
            pd.to_datetime(frame["retrieved_at_utc"], utc=True, errors="raise").max()
            for frame in frames.values()
            if not frame.empty
        ),
        default=pd.NaT,
    )
    if odds.empty:
        reasons.append("market_odds.csv contains no quotes")
        return {
            "status": "DATA QUALITY FAILURE",
            "analysis_as_of": as_of.isoformat(),
            "latest_data_timestamp": None if pd.isna(latest_data) else latest_data.isoformat(),
            "season": season,
            "week": week,
            "model_version": model_version,
            "datasets": datasets,
            "unmatched_join_rows": unmatched,
            "games_evaluated": 0,
            "markets_evaluated": 0,
            "qualifying_count": 0,
            "blocking_reasons": reasons,
        }

    odds = odds.assign(
        retrieved_dt=pd.to_datetime(odds["retrieved_at_utc"], utc=True, errors="raise"),
        quote_dt=pd.to_datetime(odds["last_update_utc"], utc=True, errors="raise"),
    )
    latest_snapshot = (
        odds[["snapshot_id", "retrieved_dt"]]
        .drop_duplicates()
        .sort_values(["retrieved_dt", "snapshot_id"], kind="stable")
        .iloc[-1]
    )
    snapshot_id = str(latest_snapshot["snapshot_id"])
    current_odds = odds.loc[odds["snapshot_id"].eq(snapshot_id)].copy()
    paired = current_odds.groupby(
        [
            "provider_event_id",
            "bookmaker_key",
            "market",
            "canonical_line",
        ],
        dropna=False,
    )["selection"].nunique()
    if bool((paired != 2).any()):
        raise RuntimeError("DATA QUALITY FAILURE: latest snapshot has unpaired market quotes")
    numeric = current_odds[
        ["american_price", "implied_probability", "vig_free_probability", "overround"]
    ].to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise RuntimeError("DATA QUALITY FAILURE: latest snapshot has non-finite market values")

    database = root / "data" / "runtime" / "nfl_bets.sqlite3"
    with sqlite3.connect(database) as connection:
        purposes = pd.read_sql_query(
            "SELECT snapshot_id,snapshot_purpose,retrieved_at_utc FROM raw_snapshots",
            connection,
        )
    snapshot_meta = purposes.loc[purposes["snapshot_id"].eq(snapshot_id)]
    if len(snapshot_meta) != 1:
        raise RuntimeError("DATA QUALITY FAILURE: latest snapshot provenance is missing")
    purpose = str(snapshot_meta.iloc[0]["snapshot_purpose"])
    if purpose != "DECISION":
        reasons.append(f"latest market snapshot is {purpose}, not DECISION")

    latest_quote = current_odds["quote_dt"].max()
    if as_of - latest_quote.to_pydatetime() > pd.Timedelta(minutes=30):
        reasons.append("latest market quote is older than 30 minutes")

    market_games = current_odds.merge(
        games[["game_id", "season", "week", "kickoff_utc"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )
    market_games["kickoff_dt"] = pd.to_datetime(
        market_games["kickoff_utc"], utc=True, errors="raise"
    )
    upcoming = market_games.loc[market_games["kickoff_dt"] > as_of]
    upcoming_weeks = sorted(upcoming["week"].dropna().astype(int).unique().tolist())
    if len(upcoming_weeks) == 1 and upcoming_weeks[0] != week:
        reasons.append(
            f"PROJECT_STATE current week {week} does not match upcoming market week "
            f"{upcoming_weeks[0]}"
        )
    market_seasons = sorted(market_games["season"].dropna().astype(int).unique().tolist())
    if market_seasons != [season]:
        reasons.append(
            f"PROJECT_STATE current season {season} does not match market seasons {market_seasons}"
        )

    current_predictions = predictions.loc[predictions["snapshot_id"].eq(snapshot_id)]
    if current_predictions.empty:
        reasons.append("no model predictions exist for the latest market snapshot")
    elif current_predictions["uncertainty_status"].eq("UNAVAILABLE_FOR_BETTING").any():
        reasons.append("model uncertainty is unavailable for betting")

    return {
        "status": "DATA QUALITY FAILURE" if reasons else "READY",
        "analysis_as_of": as_of.isoformat(),
        "latest_data_timestamp": None if pd.isna(latest_data) else latest_data.isoformat(),
        "latest_market_timestamp": latest_snapshot["retrieved_dt"].isoformat(),
        "latest_quote_timestamp": latest_quote.isoformat(),
        "latest_snapshot_id": snapshot_id,
        "latest_snapshot_purpose": purpose,
        "season": season,
        "week": week,
        "model_version": model_version,
        "datasets": datasets,
        "unmatched_join_rows": unmatched,
        "games_evaluated": int(current_odds["game_id"].nunique()),
        "markets_evaluated": int(current_odds[["game_id", "market"]].drop_duplicates().shape[0]),
        "qualifying_count": 0,
        "blocking_reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit current NFL betting readiness.")
    parser.add_argument("--as-of", required=True, help="Timezone-aware ISO-8601 timestamp")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    root = args.root.resolve()
    validate_all(Settings.for_root(root))
    report = audit_repository(root, as_of)
    if report["blocking_reasons"]:
        print("DATA QUALITY FAILURE: " + "; ".join(report["blocking_reasons"]))
    print(json.dumps(report, indent=2, default=str))
    return 2 if report["blocking_reasons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
