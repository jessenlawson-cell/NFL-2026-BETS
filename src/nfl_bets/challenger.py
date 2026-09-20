from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.features.build import materialize_feature_configuration, validate_lag_boundaries
from nfl_bets.features.challenger import CHALLENGER_METRICS
from nfl_bets.features.v11 import INJURY_DIAGNOSTICS
from nfl_bets.model.challenger import (
    MODEL_TEAM_METRICS,
    ChallengerCandidate,
    build_matchup_frame,
    materialize_opponent_adjusted,
    quantitative_drivers,
    score_challenger_contract,
)
from nfl_bets.prospective import (
    ProspectiveDataError,
    _anchor_contract,
    _append_ledger,
    _parse_utc,
    _stable_id,
    _validate_latest_completed_game,
)
from nfl_bets.util import atomic_write_text, canonical_hash, iso_utc, sha256_bytes


@dataclass(frozen=True)
class ChallengerBundle:
    candidate: ChallengerCandidate
    policy: dict[str, Any]
    artifact_hash: str
    policy_hash: str
    git_commit: str


@dataclass(frozen=True)
class PreparedChallengerFeatures:
    lags: pl.DataFrame
    materialized: pl.DataFrame
    games: pl.DataFrame
    as_of: datetime
    input_hash: str

def rank_shadow_candidates(
    candidates: list[dict[str, Any]], *, top: int = 5
) -> list[dict[str, Any]]:
    if top < 1:
        raise ValueError("top must be positive")
    qualified: list[dict[str, Any]] = []
    for source in candidates:
        row = dict(source)
        probability = float(row["model_probability"])
        margin = float(row["probability_margin"])
        conservative = max(0.0, min(1.0, probability - margin))
        push = float(row["push_probability"])
        win = (1.0 - push) * conservative
        loss = (1.0 - push) * (1.0 - conservative)
        edge = conservative - float(row["market_fair_probability"])
        expected_roi = win * float(row["net_decimal_profit"]) - loss
        row.update(
            {
                "conservative_probability": conservative,
                "conservative_probability_edge": edge,
                "conservative_expected_roi": expected_roi,
                "shadow_label": (
                    "SHADOW_CANDIDATE"
                    if edge >= 0.02 and expected_roi > 0.0
                    else "PASS"
                ),
            }
        )
        personnel_clear = row.get("injury_data_status", "COMPLETE_UNWEIGHTED") == (
            "COMPLETE_UNWEIGHTED"
        )
        if row["shadow_label"] == "SHADOW_CANDIDATE" and personnel_clear:
            qualified.append(row)
    ordered = sorted(
        qualified,
        key=lambda row: (
            -float(row["conservative_expected_roi"]),
            -float(row["conservative_probability_edge"]),
            str(row["game_id"]),
            str(row["market"]),
            str(row["selection"]),
            str(row.get("book", "")),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in ordered:
        exposure = (str(row["game_id"]), str(row["market"]))
        if exposure in seen:
            continue
        seen.add(exposure)
        selected.append(row)
        if len(selected) == top:
            break
    return selected


def load_challenger_bundle(
    version: str,
    settings: Settings | None = None,
    *,
    verify_git: bool = True,
) -> ChallengerBundle:
    resolved = settings or get_settings()
    policy_path = resolved.manifests_dir / f"prospective_policy_{version}.json"
    manifest_path = resolved.manifests_dir / f"model_{version}_development.json"
    if not policy_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen challenger policy or manifest for {version}")
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if policy.get("model_version") != version or manifest.get("model_version") != version:
        raise ProspectiveDataError("Challenger model version mismatch")
    if policy.get("status") != "SHADOW_FROZEN_WEEK3_TO_8":
        raise ProspectiveDataError("Challenger is not frozen for Week 3 through Week 8")
    artifact_path = resolved.root / str(manifest["artifact"])
    metadata_path = artifact_path.parent / "metadata.json"
    if not artifact_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Frozen challenger files are missing for {version}")
    artifact_hash = sha256_bytes(artifact_path.read_bytes())
    metadata_hash = sha256_bytes(metadata_path.read_bytes())
    spec_hash = sha256_bytes((resolved.root / "CHALLENGER_SPEC.md").read_bytes())
    for key, actual in (
        ("artifact_hash", artifact_hash),
        ("metadata_hash", metadata_hash),
        ("spec_hash", spec_hash),
    ):
        if manifest.get(key) != actual or policy.get(key) != actual:
            raise ProspectiveDataError(f"Challenger hash mismatch for {key}")
    candidate = joblib.load(artifact_path)
    if not isinstance(candidate, ChallengerCandidate):
        raise ProspectiveDataError("Frozen challenger artifact has an unexpected type")
    if candidate.version != version:
        raise ProspectiveDataError("Challenger artifact version mismatch")
    if candidate.feature_hash != policy["development_feature_hash"]:
        raise ProspectiveDataError("Challenger development feature hash mismatch")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if verify_git:
        if git_commit != policy["freeze_commit"]:
            raise ProspectiveDataError("Current Git commit differs from challenger freeze commit")
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src/nfl_bets/challenger.py",
                "src/nfl_bets/model/challenger.py",
                "src/nfl_bets/features/challenger.py",
                "CHALLENGER_SPEC.md",
            ],
            cwd=resolved.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise ProspectiveDataError("Tracked challenger code or specification is dirty")
    return ChallengerBundle(
        candidate=candidate,
        policy=policy,
        artifact_hash=artifact_hash,
        policy_hash=sha256_bytes(policy_bytes),
        git_commit=git_commit,
    )


def _prepare_challenger_features(
    bundle: ChallengerBundle, settings: Settings
) -> PreparedChallengerFeatures:
    manifest_path = settings.manifests_dir / "challenger_feature_inputs.latest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Build current challenger features before predicting")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_path = settings.root / str(manifest.get("artifact", ""))
    if not feature_path.exists():
        raise FileNotFoundError("Current challenger feature store is missing")
    input_hash = sha256_bytes(feature_path.read_bytes())
    if input_hash != manifest.get("content_hash"):
        raise ProspectiveDataError("Current challenger feature store hash mismatch")
    if manifest.get("injury_model_status") != "DIAGNOSTIC_ONLY_UNWEIGHTED":
        raise ProspectiveDataError("Unexpected challenger injury policy")
    as_of = _parse_utc(manifest.get("as_of_utc"), "challenger feature as-of")
    lags = pl.read_parquet(feature_path)
    validate_lag_boundaries(lags)
    base_materialized = materialize_feature_configuration(
        lags,
        half_life=bundle.candidate.selected_half_life,
        prior_strength=bundle.candidate.selected_prior_strength,
        metrics=CHALLENGER_METRICS,
    )
    materialized = materialize_opponent_adjusted(
        lags,
        base_materialized,
        half_life=bundle.candidate.selected_half_life,
        prior_strength=bundle.candidate.selected_prior_strength,
    )
    games = pl.read_csv(settings.root / "games.csv", infer_schema_length=100_000)
    return PreparedChallengerFeatures(lags, materialized, games, as_of, input_hash)


def _challenger_game_features(
    game_id: str,
    snapshot_time: datetime,
    prepared: PreparedChallengerFeatures,
    candidate: ChallengerCandidate,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if prepared.as_of >= snapshot_time:
        raise ProspectiveDataError("Challenger feature as-of must predate the odds snapshot")
    selected_lags = prepared.lags.filter(pl.col("game_id") == game_id)
    selected = prepared.materialized.filter(pl.col("game_id") == game_id)
    game = prepared.games.filter(pl.col("game_id") == game_id)
    if selected.height != 2 or selected_lags.height != 2 or game.height != 1:
        raise ProspectiveDataError("Challenger feature/game rows are missing or duplicated")
    validate_lag_boundaries(selected_lags)
    target_kickoff = _parse_utc(game["kickoff_utc"][0], "game kickoff")
    _validate_latest_completed_game(
        selected_lags, prepared.games, target_kickoff, prepared.as_of
    )
    matrix = build_matchup_frame(selected, game).select(candidate.feature_columns)
    if matrix.height != 1 or matrix.null_count().row(0) != tuple(0 for _ in matrix.columns):
        raise ProspectiveDataError("Challenger pregame model features are incomplete")
    hash_columns = [
        "game_id",
        "team_id",
        "kickoff_dt",
        *[
            f"lag{offset}_{metric}"
            for offset in range(1, 5)
            for metric in CHALLENGER_METRICS
        ],
    ]
    row_hash = sha256_bytes(
        selected_lags.select(hash_columns)
        .sort(["game_id", "team_id"])
        .write_csv()
        .encode()
    )
    injuries: dict[str, Any] = {}
    for row in selected_lags.select("team_id", "is_home", *INJURY_DIAGNOSTICS).to_dicts():
        side = "home" if bool(row["is_home"]) else "away"
        injuries[f"{side}_injuries"] = {
            "team_id": row["team_id"],
            **{name: row.get(name) for name in INJURY_DIAGNOSTICS},
        }
    injuries["injury_data_status"] = (
        "COMPLETE_UNWEIGHTED"
        if all(
            value is not None
            for side in ("home_injuries", "away_injuries")
            for name, value in injuries.get(side, {}).items()
            if name != "team_id"
        )
        else "INCOMPLETE_UNWEIGHTED"
    )
    return matrix.to_numpy(), row_hash, injuries


def _book_candidates(
    *,
    game_id: str,
    game_label: str,
    market: str,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    candidate: ChallengerCandidate,
    injuries: dict[str, Any],
    snapshot_time: datetime,
    quote_max_age_minutes: int,
) -> list[dict[str, Any]]:
    orientation = "HOME" if market == "spreads" else "OVER"
    opposite = "AWAY" if market == "spreads" else "UNDER"
    results: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        line = row.get("canonical_line")
        if line is not None:
            grouped.setdefault((str(row["bookmaker_key"]), float(line)), []).append(row)
    for (book, line), quotes in sorted(grouped.items()):
        by_selection = {str(row["selection"]): row for row in quotes}
        if set(by_selection) != {orientation, opposite} or len(quotes) != 2:
            continue
        updated = [
            _parse_utc(
                row.get("source_updated_at_utc") or row.get("last_update_utc"),
                "book quote update",
            )
            for row in quotes
        ]
        if any(value > snapshot_time for value in updated) or (
            snapshot_time - min(updated) > timedelta(minutes=quote_max_age_minutes)
        ):
            continue
        if any(float(row["overround"]) <= 1.0 for row in quotes):
            continue
        if not math.isclose(
            sum(float(row["vig_free_probability"]) for row in quotes),
            1.0,
            abs_tol=1e-9,
        ):
            continue
        scores = score_challenger_contract(candidate, market, line, features)
        drivers = quantitative_drivers(candidate, market, features)
        conditional = float(scores["calibrated_non_push_win_probability"])
        push = float(scores["model_push_probability"])
        for selection in (orientation, opposite):
            quote = by_selection[selection]
            price = int(quote["american_price"])
            net_profit = price / 100.0 if price > 0 else 100.0 / abs(price)
            model_probability = conditional if selection == orientation else 1.0 - conditional
            results.append(
                {
                    "game_id": game_id,
                    "game": game_label,
                    "market": market,
                    "selection": selection,
                    "book": book,
                    "line": line,
                    "offered_point": float(quote["point"]),
                    "price": price,
                    "net_decimal_profit": net_profit,
                    "break_even_probability": 1.0 / (1.0 + net_profit),
                    "raw_football_projection": scores["raw_football_projection"],
                    "market_regressed_projection": None,
                    "final_projection": scores["final_projection"],
                    "model_probability": model_probability,
                    "push_probability": push,
                    "market_fair_probability": float(quote["vig_free_probability"]),
                    "probability_edge": model_probability
                    - float(quote["vig_free_probability"]),
                    "expected_roi": (1.0 - push)
                    * (model_probability * net_profit - (1.0 - model_probability)),
                    "probability_margin": scores["probability_margin"],
                    "key_number_effect": (
                        "EMPIRICAL_SIGNED_KEY_MASS" if market == "spreads" else "NOT_APPLICABLE"
                    ),
                    "market_timeline": "CURRENT_DECISION_SNAPSHOT; CLOSE_PENDING",
                    "correlation": "ONE_SELECTION_PER_GAME_MARKET",
                    "kelly": None,
                    "fractional_kelly": None,
                    "final_stake": None,
                    "decision": "PASS",
                    "what_would_invalidate": (
                        "stale or corrected inputs, lineup uncertainty, price movement, "
                        "or failure of prospective calibration"
                    ),
                    "top_quantitative_drivers": drivers,
                    **injuries,
                }
            )
    return results


def _power_rankings(
    prepared: PreparedChallengerFeatures, candidate: ChallengerCandidate
) -> list[dict[str, Any]]:
    latest = (
        prepared.materialized.filter(pl.col("kickoff_dt") > pl.lit(prepared.as_of))
        .sort(["team_id", "kickoff_dt", "game_id"])
        .group_by("team_id", maintain_order=True)
        .first()
    )
    if latest.is_empty():
        return []
    averages = latest.select(
        *[pl.col(metric).mean().alias(metric) for metric in MODEL_TEAM_METRICS]
    ).row(0, named=True)
    rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    for team in latest.iter_rows(named=True):
        game_id = f"neutral_{team['team_id']}"
        rows.append(
            {
                "game_id": game_id,
                "is_home": True,
                **{metric: team[metric] for metric in MODEL_TEAM_METRICS},
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "is_home": False,
                **{metric: averages[metric] for metric in MODEL_TEAM_METRICS},
            }
        )
        game_rows.append(
            {
                "game_id": game_id,
                "team_id": str(team["team_id"]),
                "home_rest": 7.0,
                "away_rest": 7.0,
                "roof": "outdoors",
                "surface": "grass",
            }
        )
    games = pl.DataFrame(game_rows)
    matrix = build_matchup_frame(pl.DataFrame(rows), games).select(
        "game_id", "team_id", *candidate.feature_columns
    )
    x = matrix.select(candidate.feature_columns).to_numpy()
    margin = np.asarray(candidate.spread_model.predict(x), dtype=float)
    total = np.asarray(candidate.total_model.predict(x), dtype=float)
    centered_margin = margin - float(np.mean(margin))
    centered_total = total - float(np.mean(total))
    rankings = [
        {
            "team_id": team_id,
            "neutral_field_rating": float(centered_margin[index]),
            "total_environment_rating": float(centered_total[index]),
        }
        for index, team_id in enumerate(matrix["team_id"].to_list())
    ]
    ordered = sorted(
        rankings,
        key=lambda row: (-float(row["neutral_field_rating"]), str(row["team_id"])),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def predict_challenger_snapshot(
    snapshot_id: str,
    *,
    version: str = "challenger-0.1.0",
    top: int = 5,
    settings: Settings | None = None,
    prediction_time: datetime | None = None,
    verify_git: bool = True,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    now = (prediction_time or datetime.now(UTC)).astimezone(UTC)
    bundle = load_challenger_bundle(version, resolved, verify_git=verify_git)
    prepared = _prepare_challenger_features(bundle, resolved)
    with connect(resolved) as connection:
        snapshot = connection.execute(
            "SELECT * FROM raw_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        request = connection.execute(
            "SELECT status FROM api_requests WHERE raw_snapshot_id=?", (snapshot_id,)
        ).fetchone()
        quotes = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM market_odds WHERE snapshot_id=? "
                "ORDER BY game_id,market,bookmaker_key,selection",
                (snapshot_id,),
            )
        ]
        consensus = {
            (str(row["provider_event_id"]), str(row["market"])): dict(row)
            for row in connection.execute(
                "SELECT * FROM market_consensus WHERE snapshot_id=?", (snapshot_id,)
            )
        }
        games = {
            str(row["game_id"]): dict(row)
            for row in connection.execute("SELECT * FROM games")
        }
    if snapshot is None or snapshot["snapshot_purpose"] != "DECISION":
        raise ProspectiveDataError(
            "Only an immutable DECISION snapshot can create challenger predictions"
        )
    if request is None or request["status"] != "COMPLETE":
        raise ProspectiveDataError("Challenger prediction requires a completely parsed snapshot")
    snapshot_time = _parse_utc(snapshot["retrieved_at_utc"], "snapshot retrieval time")
    if now < snapshot_time:
        raise ProspectiveDataError("Prediction time cannot precede snapshot retrieval")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for quote in quotes:
        grouped.setdefault(
            (str(quote["provider_event_id"]), str(quote["game_id"]), str(quote["market"])),
            [],
        ).append(quote)
    records: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    feature_cache: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {}
    for (event_id, game_id, market), market_rows in sorted(grouped.items()):
        game = games.get(game_id)
        if game is None:
            raise ProspectiveDataError(f"Snapshot references unknown game {game_id}")
        kickoff = _parse_utc(game["kickoff_utc"], "game kickoff")
        if now >= kickoff:
            continue
        week = int(game["week"])
        if not int(bundle.policy["prospective_start_week"]) <= week <= int(
            bundle.policy["prospective_end_week"]
        ):
            continue
        contract = _anchor_contract(
            market_rows,
            market,
            snapshot_time,
            int(bundle.policy["quote_max_age_minutes"]),
        )
        feature_error: str | None = None
        try:
            if game_id not in feature_cache:
                feature_cache[game_id] = _challenger_game_features(
                    game_id, snapshot_time, prepared, bundle.candidate
                )
            features, row_hash, injuries = feature_cache[game_id]
        except (ValueError, ProspectiveDataError) as exc:
            feature_error = str(exc)
            features = None
            row_hash = None
            injuries = {"injury_data_status": "UNAVAILABLE"}
        scores: dict[str, float] = {}
        if contract.valid and contract.line is not None and features is not None:
            scores = score_challenger_contract(
                bundle.candidate, market, contract.line, features
            )
            candidate_rows.extend(
                _book_candidates(
                    game_id=game_id,
                    game_label=f"{game['away_team']} at {game['home_team']}",
                    market=market,
                    rows=market_rows,
                    features=features,
                    candidate=bundle.candidate,
                    injuries=injuries,
                    snapshot_time=snapshot_time,
                    quote_max_age_minutes=int(bundle.policy["quote_max_age_minutes"]),
                )
            )
        status = (
            "SHADOW_ELIGIBLE"
            if contract.valid and features is not None
            else f"INELIGIBLE_{contract.reason or 'FEATURES'}"
        )
        consensus_row = consensus.get((event_id, market), {})
        created = iso_utc(now)
        record: dict[str, Any] = {
            "prediction_id": _stable_id(version, snapshot_id, game_id, market),
            "model_version": version,
            "model_artifact_hash": bundle.artifact_hash,
            "model_spec_hash": bundle.candidate.spec_hash,
            "policy_hash": bundle.policy_hash,
            "git_commit": bundle.git_commit,
            "snapshot_id": snapshot_id,
            "provider_event_id": event_id,
            "game_id": game_id,
            "season": int(game["season"]),
            "week": week,
            "kickoff_utc": iso_utc(kickoff),
            "market": market,
            "orientation": "HOME" if market == "spreads" else "OVER",
            "prediction_created_at_utc": created,
            "snapshot_retrieved_at_utc": iso_utc(snapshot_time),
            "feature_as_of_utc": iso_utc(prepared.as_of),
            "feature_input_hash": prepared.input_hash,
            "feature_row_hash": row_hash,
            "pinnacle_updated_at_utc": (
                iso_utc(contract.updated_at) if contract.updated_at else None
            ),
            "pinnacle_line": contract.line,
            "pinnacle_orientation_price": contract.orientation_price,
            "pinnacle_other_price": contract.other_price,
            "pinnacle_orientation_no_vig_probability": contract.orientation_probability,
            "pinnacle_overround": contract.overround,
            "consensus_line": consensus_row.get("consensus_line"),
            "consensus_probability": consensus_row.get("consensus_probability"),
            "consensus_status": consensus_row.get("status", "UNAVAILABLE"),
            "retail_books_count": int(consensus_row.get("retail_books_count", 0)),
            "raw_adjustment": scores.get("raw_adjustment"),
            "adjustment_weight": scores.get("adjustment_weight"),
            "final_projection": scores.get("final_projection"),
            "raw_non_push_win_probability": scores.get("raw_non_push_win_probability"),
            "calibrated_non_push_win_probability": scores.get(
                "calibrated_non_push_win_probability"
            ),
            "model_win_probability": scores.get("model_win_probability"),
            "model_push_probability": scores.get("model_push_probability"),
            "model_loss_probability": scores.get("model_loss_probability"),
            "uncertainty_status": "MODELED_DEVELOPMENT_MARGIN_SHADOW_ONLY",
            "eligibility_status": status,
            "decision": "PASS",
            "pass_reason": ";".join(
                value
                for value in (
                    "EXPERIMENTAL_SHADOW_NO_STAKES",
                    status,
                    feature_error,
                )
                if value
            ),
            "source": "local:challenger+the-odds-api",
            "retrieved_at_utc": created,
            "source_updated_at_utc": (
                iso_utc(contract.updated_at) if contract.updated_at else None
            ),
            "schema_version": resolved.schema_version,
        }
        record["content_hash"] = canonical_hash(record)
        records.append(record)
    inserted, unchanged = _append_ledger(
        "model_predictions",
        records,
        resolved,
        transient={"prediction_created_at_utc", "retrieved_at_utc"},
        order_by="season,week,kickoff_utc,game_id,market,snapshot_retrieved_at_utc,prediction_id",
    )
    ranked = rank_shadow_candidates(candidate_rows, top=top)
    power_rankings = _power_rankings(prepared, bundle.candidate)
    report = {
        "mode": "EXPERIMENTAL_SHADOW_NO_STAKES",
        "model_version": version,
        "snapshot_id": snapshot_id,
        "snapshot_purpose": "DECISION",
        "snapshot_time": iso_utc(snapshot_time),
        "prediction_time": iso_utc(now),
        "feature_as_of": iso_utc(prepared.as_of),
        "inserted": inserted,
        "unchanged": unchanged,
        "games_evaluated": len(feature_cache),
        "markets_recorded": len(records),
        "qualifying_count": len(ranked),
        "injury_warning": "Injury diagnostics are unweighted and do not alter probabilities.",
        "power_rankings": power_rankings,
        "decision": "PASS",
        "top": ranked,
    }
    report_path = resolved.reports_dir / f"challenger_{version}_{snapshot_id}.json"
    atomic_write_text(report_path, json.dumps(report, sort_keys=True, indent=2))
    report["report_path"] = str(report_path.relative_to(resolved.root)).replace("\\", "/")
    return report


def compare_challenger(
    *,
    through_week: int,
    version: str = "challenger-0.1.0",
    baseline: str = "1.1.2",
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    with connect(resolved) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM prospective_evaluations WHERE season=2026 AND week<=? "
            "AND model_version IN (?,?)",
            connection,
            params=(through_week, version, baseline),
        )
    keys = ["game_id", "market", "decision_snapshot_id"]
    challenger = frame.loc[frame["model_version"].eq(version)].copy()
    frozen = frame.loc[frame["model_version"].eq(baseline)].copy()
    matched = challenger.merge(
        frozen,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_challenger", "_baseline"),
    )

    def summary(data: pd.DataFrame, suffix: str) -> dict[str, Any]:
        eligible = data.loc[data[f"eligible_non_push_{suffix}"].eq(1)].copy()
        if eligible.empty:
            return {"eligible_non_push": 0}
        projection = pd.to_numeric(
            eligible[f"projection_error_{suffix}"], errors="coerce"
        ).dropna()
        market_projection = pd.to_numeric(
            eligible[f"market_projection_error_{suffix}"], errors="coerce"
        ).dropna()

        def mean_or_none(column: str) -> float | None:
            values = pd.to_numeric(eligible[column], errors="coerce").dropna()
            return None if values.empty else float(values.mean())

        return {
            "eligible_non_push": int(len(eligible)),
            "brier": mean_or_none(f"model_brier_{suffix}"),
            "log_loss": mean_or_none(f"model_log_loss_{suffix}"),
            "projection_rmse": (
                float(np.sqrt(np.mean(np.square(projection)))) if not projection.empty else None
            ),
            "market_projection_rmse": (
                float(np.sqrt(np.mean(np.square(market_projection))))
                if not market_projection.empty
                else None
            ),
            "mean_probability_clv": mean_or_none(
                f"market_probability_movement_{suffix}"
            ),
        }

    by_market: dict[str, Any] = {}
    for market in ("spreads", "totals"):
        selected = matched.loc[matched["market"].eq(market)]
        by_market[market] = {
            "matched_contracts": int(len(selected)),
            "challenger": summary(selected, "challenger"),
            "baseline": summary(selected, "baseline"),
        }
    report = {
        "challenger_version": version,
        "baseline_version": baseline,
        "season": 2026,
        "through_week": through_week,
        "status": (
            "MATCHED_EVIDENCE_AVAILABLE"
            if not matched.empty
            else "INSUFFICIENT_MATCHED_EVIDENCE"
        ),
        "challenger_evaluations": int(len(challenger)),
        "baseline_evaluations": int(len(frozen)),
        "matched_evaluations": int(len(matched)),
        "unmatched_challenger": int(len(challenger) - len(matched)),
        "unmatched_baseline": int(len(frozen) - len(matched)),
        "markets": by_market,
        "promotion_authorized": False,
    }
    path = (
        resolved.reports_dir
        / f"challenger_comparison_{version}_through_week_{through_week}.json"
    )
    atomic_write_text(path, json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    report["report_path"] = str(path.relative_to(resolved.root)).replace("\\", "/")
    return report
