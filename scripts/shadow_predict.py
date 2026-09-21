from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.prospective import (
    ProspectiveDataError,
    _game_feature_vector,
    _prepare_features,
    _score_contract,
    load_frozen_bundle,
)

INJURY_FIELDS = (
    "qb_out_doubtful",
    "ol_out_doubtful",
    "defensive_front_out_doubtful",
    "secondary_out_doubtful",
    "questionable_players",
)


def football_evidence(
    market: str, selection: str, raw_adjustment: float, adjustment_weight: float
) -> dict[str, Any]:
    weighted = raw_adjustment * adjustment_weight
    if adjustment_weight == 0.0:
        return {
            "weighted_adjustment": 0.0,
            "signal_source": "MARKET_CALIBRATION_ONLY",
            "football_direction": "NEUTRAL",
        }
    positive_selection = "HOME" if market == "spreads" else "OVER"
    negative_selection = "AWAY" if market == "spreads" else "UNDER"
    if selection not in {positive_selection, negative_selection}:
        raise ValueError(f"invalid {market} selection: {selection}")
    supports = (weighted > 0.0 and selection == positive_selection) or (
        weighted < 0.0 and selection == negative_selection
    )
    return {
        "weighted_adjustment": weighted,
        "signal_source": "FOOTBALL_ADJUSTED",
        "football_direction": (
            "NEUTRAL"
            if weighted == 0.0
            else "SUPPORTS_SELECTION"
            if supports
            else "OPPOSES_SELECTION"
        ),
    }


def injury_overlay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2 or {bool(row["is_home"]) for row in rows} != {False, True}:
        raise ValueError("injury diagnostics require one home and one away team row")
    missing = any(row.get(field) is None for row in rows for field in INJURY_FIELDS)

    def team(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "team_id": str(row["team_id"]),
            **{
                field: None if row.get(field) is None else int(row[field])
                for field in INJURY_FIELDS
            },
        }

    away = next(row for row in rows if not bool(row["is_home"]))
    home = next(row for row in rows if bool(row["is_home"]))
    return {
        "injury_data_status": ("INCOMPLETE_UNWEIGHTED" if missing else "COMPLETE_UNWEIGHTED"),
        "away_injuries": team(away),
        "home_injuries": team(home),
    }


def evidence_summary(candidates: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "signal_source_counts": {
            source: sum(row["signal_source"] == source for row in candidates)
            for source in ("MARKET_CALIBRATION_ONLY", "FOOTBALL_ADJUSTED")
        },
        "football_direction_counts": {
            direction: sum(row["football_direction"] == direction for row in candidates)
            for direction in ("SUPPORTS_SELECTION", "OPPOSES_SELECTION", "NEUTRAL")
        },
    }


def contract_candidates(
    *,
    game_id: str,
    game: str,
    market: str,
    line: float,
    book: str,
    scores: dict[str, float],
    quotes: list[dict[str, Any]],
    injuries: dict[str, Any],
    feature_as_of: str,
) -> list[dict[str, Any]]:
    orientation = "HOME" if market == "spreads" else "OVER"
    other = "AWAY" if market == "spreads" else "UNDER"
    by_selection = {str(row["selection"]): row for row in quotes}
    if set(by_selection) != {orientation, other}:
        raise ValueError("two-sided contract selections are invalid")
    non_push = float(scores["calibrated_non_push_win_probability"])
    push = float(scores["model_push_probability"])
    outcomes = {
        orientation: (
            non_push,
            float(scores["model_win_probability"]),
            float(scores["model_loss_probability"]),
        ),
        other: (
            1.0 - non_push,
            float(scores["model_loss_probability"]),
            float(scores["model_win_probability"]),
        ),
    }
    candidates: list[dict[str, Any]] = []
    for selection in (orientation, other):
        row = by_selection[selection]
        price = int(row["american_price"])
        if price == 0:
            raise ValueError("American odds cannot be zero")
        net_profit = price / 100.0 if price > 0 else 100.0 / abs(price)
        model_probability, win_probability, loss_probability = outcomes[selection]
        market_probability = float(row["vig_free_probability"])
        candidates.append(
            {
                "game_id": game_id,
                "game": game,
                "market": market,
                "selection": selection,
                "line": line,
                "offered_point": float(row["point"]),
                "price": price,
                "book": book,
                "feature_as_of": feature_as_of,
                **injuries,
                "raw_adjustment": float(scores["raw_adjustment"]),
                "adjustment_weight": float(scores["adjustment_weight"]),
                **football_evidence(
                    market,
                    selection,
                    float(scores["raw_adjustment"]),
                    float(scores["adjustment_weight"]),
                ),
                "final_projection": float(scores["final_projection"]),
                "model_probability": model_probability,
                "model_win_probability": win_probability,
                "push_probability": push,
                "model_loss_probability": loss_probability,
                "market_fair_probability": market_probability,
                "break_even_probability": 1.0 / (1.0 + net_profit),
                "probability_edge": model_probability - market_probability,
                "expected_roi": win_probability * net_profit - loss_probability,
            }
        )
    return candidates


def shadow_predictions(
    root: Path,
    snapshot_id: str,
    as_of: datetime,
    season: int,
    week: int,
    version: str = "1.1.2",
) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise ValueError("analysis as-of timestamp must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    settings = Settings.for_root(root)
    initialize_database(settings)
    bundle = load_frozen_bundle(version, settings)
    prepared = _prepare_features(bundle, settings)
    with connect(settings) as connection:
        snapshot = connection.execute(
            "SELECT * FROM raw_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        request = connection.execute(
            "SELECT status FROM api_requests WHERE raw_snapshot_id=?", (snapshot_id,)
        ).fetchone()
    if snapshot is None:
        raise ProspectiveDataError(f"Raw snapshot {snapshot_id} does not exist")
    purpose = str(snapshot["snapshot_purpose"])
    if purpose not in {"DECISION", "DIAGNOSTIC"}:
        raise ProspectiveDataError("Shadow scoring requires a DECISION or DIAGNOSTIC snapshot")
    if request is None or request["status"] != "COMPLETE":
        raise ProspectiveDataError("Shadow scoring requires a completely parsed snapshot")
    snapshot_time = datetime.fromisoformat(
        str(snapshot["retrieved_at_utc"]).replace("Z", "+00:00")
    ).astimezone(UTC)
    if as_of < snapshot_time or as_of - snapshot_time > timedelta(minutes=30):
        raise ProspectiveDataError("Shadow scoring must run within 30 minutes of capture")

    games = pd.read_csv(root / "games.csv", low_memory=False)
    odds = pd.read_csv(root / "market_odds.csv", low_memory=False)
    games = games.loc[games["season"].eq(season) & games["week"].eq(week)].copy()
    games["kickoff_dt"] = pd.to_datetime(games["kickoff_utc"], utc=True, errors="raise")
    odds = odds.loc[odds["snapshot_id"].eq(snapshot_id)].merge(
        games[["game_id", "away_team", "home_team", "kickoff_dt"]],
        on="game_id",
        how="inner",
        validate="many_to_one",
    )
    odds = odds.loc[odds["kickoff_dt"] > as_of].copy()
    if odds.empty:
        raise ProspectiveDataError("Snapshot contains no unstarted games for the requested week")

    all_candidates: list[dict[str, Any]] = []
    skipped_contracts = 0
    score_cache: dict[tuple[str, str, float], dict[str, float]] = {}
    injury_cache: dict[str, dict[str, Any]] = {}
    group_columns = ["game_id", "bookmaker_key", "market", "canonical_line"]
    for key, frame in odds.groupby(group_columns, dropna=False, sort=True):
        game_id, book, market, line_value = key
        if pd.isna(line_value) or market not in {"spreads", "totals"}:
            skipped_contracts += 1
            continue
        expected = {"HOME", "AWAY"} if market == "spreads" else {"OVER", "UNDER"}
        if set(frame["selection"].astype(str)) != expected or len(frame) != 2:
            skipped_contracts += 1
            continue
        updates = pd.to_datetime(
            frame["source_updated_at_utc"].fillna(frame["last_update_utc"]),
            utc=True,
            errors="raise",
        )
        stale = snapshot_time - updates.min().to_pydatetime() > timedelta(minutes=30)
        if bool((updates > snapshot_time).any()) or stale:
            skipped_contracts += 1
            continue
        overround = frame["overround"].astype(float)
        fair_sum = frame["vig_free_probability"].astype(float).sum()
        if bool((overround <= 1.0).any()) or not math.isclose(fair_sum, 1.0, abs_tol=1e-9):
            skipped_contracts += 1
            continue
        line = float(line_value)
        score_key = (str(game_id), str(market), line)
        game_key = str(game_id)
        if game_key not in injury_cache:
            injury_cache[game_key] = injury_overlay(
                prepared.lags.filter(prepared.lags["game_id"] == game_key)
                .select("team_id", "is_home", *INJURY_FIELDS)
                .to_dicts()
            )
        if score_key not in score_cache:
            features, _ = _game_feature_vector(str(game_id), snapshot_time, prepared)
            score_cache[score_key] = _score_contract(bundle.candidate, str(market), line, features)
        first = frame.iloc[0]
        all_candidates.extend(
            contract_candidates(
                game_id=str(game_id),
                game=f"{first['away_team']} at {first['home_team']}",
                market=str(market),
                line=line,
                book=str(book),
                scores=score_cache[score_key],
                quotes=frame.to_dict(orient="records"),
                injuries=injury_cache[game_key],
                feature_as_of=prepared.as_of.isoformat(),
            )
        )

    ordered = sorted(
        (
            row
            for row in all_candidates
            if row["probability_edge"] >= 0.02 and row["expected_roi"] > 0.0
        ),
        key=lambda row: (
            -row["expected_roi"],
            -row["probability_edge"],
            row["game_id"],
            row["market"],
            row["selection"],
            row["book"],
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in ordered:
        exposure = (row["game_id"], row["market"])
        if exposure in seen:
            continue
        seen.add(exposure)
        selected.append(row)
    return {
        "mode": "EXPERIMENTAL_SHADOW_NO_STAKES",
        "snapshot_id": snapshot_id,
        "snapshot_purpose": purpose,
        "snapshot_time": snapshot_time.isoformat(),
        "analysis_as_of": as_of.isoformat(),
        "feature_as_of": prepared.as_of.isoformat(),
        "season": season,
        "week": week,
        "model_version": version,
        "games_evaluated": int(odds["game_id"].nunique()),
        "contracts_evaluated": len(score_cache),
        "contracts_skipped": skipped_contracts,
        "qualifying_count": len(selected),
        **evidence_summary(selected),
        "injury_warning": (
            "Injury diagnostics are an unweighted warning overlay and do not alter probabilities."
        ),
        "top": selected[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only shadow NFL market scorer.")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--as-of", required=True, help="Timezone-aware ISO-8601 timestamp")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--version", default="1.1.2")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = shadow_predictions(
        root=args.root.resolve(),
        snapshot_id=args.snapshot_id,
        as_of=datetime.fromisoformat(args.as_of.replace("Z", "+00:00")),
        season=args.season,
        week=args.week,
        version=args.version,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
