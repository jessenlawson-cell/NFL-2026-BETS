from __future__ import annotations

import html
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl

from nfl_bets.config import Settings, get_settings
from nfl_bets.db import connect, initialize_database, transaction
from nfl_bets.features.build import materialize_feature_configuration, validate_lag_boundaries
from nfl_bets.features.v11 import V11_METRICS
from nfl_bets.model.training import _append_history
from nfl_bets.model.v11 import V11_FEATURE_COLUMNS, V11Candidate, _matchup_expressions
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.util import atomic_write_text, canonical_hash, iso_utc, sha256_bytes

DEFAULT_MODEL_VERSION = "1.1.2"
ELIGIBLE_CLOSE_STATUS = "ELIGIBLE_CLOSE_PROXY"
SELECTION_RULE_VERSION = "1.0.0"
MODEL_SOURCE = "local:v11-prospective-observer"


class ProspectiveDataError(RuntimeError):
    pass


class ImmutableLedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenBundle:
    candidate: V11Candidate
    policy: dict[str, Any]
    policy_hash: str
    artifact_hash: str
    git_commit: str


@dataclass(frozen=True)
class PreparedFeatures:
    lags: pl.DataFrame
    materialized: pl.DataFrame
    games: pl.DataFrame
    as_of: datetime
    input_hash: str


@dataclass(frozen=True)
class AnchorContract:
    valid: bool
    reason: str | None
    line: float | None
    orientation_price: int | None
    other_price: int | None
    orientation_probability: float | None
    overround: float | None
    updated_at: datetime | None


def _parse_utc(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveDataError(f"Invalid {label}: {value}") from exc
    if parsed.tzinfo is None:
        raise ProspectiveDataError(f"Timezone-naive {label}: {value}")
    return parsed.astimezone(UTC)


def _stable_id(*parts: object) -> str:
    return sha256_bytes("|".join(str(part) for part in parts).encode("utf-8"))


def _run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_git_identity(settings: Settings, policy: dict[str, Any]) -> str:
    try:
        commit = _run_git(settings.root, "rev-parse", "HEAD")
        tagged_commit = _run_git(settings.root, "rev-list", "-n", "1", str(policy["freeze_tag"]))
        if tagged_commit != policy["freeze_commit"]:
            raise ProspectiveDataError(
                "Prospective freeze tag does not resolve to its policy commit"
            )
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={settings.root}",
                "merge-base",
                "--is-ancestor",
                str(policy["freeze_commit"]),
                commit,
            ],
            cwd=settings.root,
            check=True,
            capture_output=True,
        )
        status = _run_git(
            settings.root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "pyproject.toml",
            "requirements.txt",
            "requirements.lock",
            "Dockerfile",
            "docker-compose.yml",
            "MODEL_SPEC.md",
            "MODEL_SPEC_V1_1.md",
            f"manifests/model_{policy['model_version']}_development.json",
            f"manifests/prospective_policy_{policy['model_version']}.json",
        )
    except (KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise ProspectiveDataError("Unable to verify the prospective Git identity") from exc
    if status:
        raise ProspectiveDataError(
            "Tracked model code or specifications are dirty; commit them before predicting"
        )
    return commit


def load_frozen_bundle(
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
    *,
    verify_git: bool = True,
) -> FrozenBundle:
    resolved = settings or get_settings()
    policy_path = resolved.manifests_dir / f"prospective_policy_{version}.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing prospective policy for model {version}")
    policy_bytes = policy_path.read_bytes()
    policy = cast(dict[str, Any], json.loads(policy_bytes))
    if policy.get("model_version") != version:
        raise ProspectiveDataError("Prospective policy model version mismatch")
    if policy.get("status") != "LOCKED_UNTESTED_2026":
        raise ProspectiveDataError("Prospective policy is not locked and untested")
    if policy.get("schema_version") != resolved.schema_version:
        raise ProspectiveDataError("Prospective policy schema version mismatch")

    manifest_path = resolved.manifests_dir / f"model_{version}_development.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen model manifest for {version}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    artifact_path = resolved.root / str(manifest["artifact"])
    metadata_path = artifact_path.parent / "metadata.json"
    invalidation_path = artifact_path.parent / "invalidation.json"
    if invalidation_path.exists():
        raise ProspectiveDataError(f"Model {version} has been invalidated")
    if not artifact_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Frozen model files are missing for {version}")
    artifact_hash = sha256_bytes(artifact_path.read_bytes())
    metadata_hash = sha256_bytes(metadata_path.read_bytes())
    spec_hash = sha256_bytes((resolved.root / "MODEL_SPEC_V1_1.md").read_bytes())
    if manifest.get("artifact_hash") != artifact_hash:
        raise ProspectiveDataError("Frozen model manifest artifact hash mismatch")
    if manifest.get("metadata_hash") != metadata_hash:
        raise ProspectiveDataError("Frozen model manifest metadata hash mismatch")
    if manifest.get("spec_hash") != spec_hash:
        raise ProspectiveDataError("Frozen model manifest specification hash mismatch")
    checks = {
        "artifact_hash": artifact_hash,
        "metadata_hash": metadata_hash,
        "spec_hash": spec_hash,
        "development_feature_hash": manifest.get("development_feature_hash"),
        "prospective_start_utc": manifest.get("prospective_start_utc"),
    }
    for key, actual in checks.items():
        if policy.get(key) != actual:
            raise ProspectiveDataError(f"Frozen model policy mismatch for {key}")
    candidate = cast(V11Candidate, joblib.load(artifact_path))
    if candidate.version != version:
        raise ProspectiveDataError("Candidate version does not match its policy")
    if candidate.feature_hash != policy["development_feature_hash"]:
        raise ProspectiveDataError("Candidate feature hash does not match its policy")
    if candidate.spec_hash != policy["spec_hash"]:
        raise ProspectiveDataError("Candidate specification hash does not match its policy")
    if candidate.prospective_start_utc != policy["prospective_start_utc"]:
        raise ProspectiveDataError("Candidate prospective cutoff does not match its policy")
    selected = cast(dict[str, Any], policy.get("selected_configuration", {}))
    candidate_settings = {
        "half_life": candidate.selected_half_life,
        "prior_strength": candidate.selected_prior_strength,
        "spread_adjustment_weight": candidate.spread_adjustment_weight,
        "spread_ridge_alpha": candidate.spread_ridge_alpha,
        "total_adjustment_weight": candidate.total_adjustment_weight,
        "total_ridge_alpha": candidate.total_ridge_alpha,
    }
    if selected != candidate_settings:
        raise ProspectiveDataError("Candidate configuration does not match its policy")
    git_commit = (
        _verify_git_identity(resolved, policy) if verify_git else str(policy["freeze_commit"])
    )
    return FrozenBundle(
        candidate=candidate,
        policy=policy,
        policy_hash=sha256_bytes(policy_bytes),
        artifact_hash=artifact_hash,
        git_commit=git_commit,
    )


def _prepare_features(bundle: FrozenBundle, settings: Settings) -> PreparedFeatures:
    manifest_path = settings.manifests_dir / "v11_feature_inputs.latest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Build current V1.1 features before predicting")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    lag_path = settings.root / str(manifest.get("artifact", ""))
    if not lag_path.exists():
        raise FileNotFoundError("The current V1.1 feature store is missing")
    input_hash = sha256_bytes(lag_path.read_bytes())
    if input_hash != manifest.get("content_hash"):
        raise ProspectiveDataError("Current V1.1 feature store hash mismatch")
    if manifest.get("injury_model_status") != "DISABLED_INSUFFICIENT_HISTORICAL_COVERAGE":
        raise ProspectiveDataError("Injury diagnostics are not safely model-disabled")
    as_of = _parse_utc(manifest.get("as_of_utc"), "feature as-of timestamp")
    lags = pl.read_parquet(lag_path)
    validate_lag_boundaries(lags)
    candidate = bundle.candidate
    materialized = materialize_feature_configuration(
        lags,
        half_life=candidate.selected_half_life,
        prior_strength=candidate.selected_prior_strength,
        metrics=V11_METRICS,
    )
    games = pl.read_csv(settings.root / "games.csv", infer_schema_length=100_000)
    return PreparedFeatures(lags, materialized, games, as_of, input_hash)


def _validate_latest_completed_game(
    selected_lags: pl.DataFrame,
    games: pl.DataFrame,
    target_kickoff: datetime,
    feature_as_of: datetime,
) -> None:
    cutoff = min(target_kickoff, feature_as_of)
    season = int(selected_lags["season"][0])
    scheduled = games.with_columns(
        pl.col("kickoff_utc").str.to_datetime(time_zone="UTC", strict=False).alias("kickoff_dt")
    )
    for team_id in selected_lags["team_id"].to_list():
        completed = scheduled.filter(
            (pl.col("season") == season)
            & (pl.col("game_type") == "REG")
            & (pl.col("kickoff_dt") < pl.lit(cutoff))
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
            & ((pl.col("home_team") == team_id) | (pl.col("away_team") == team_id))
        ).sort("kickoff_dt")
        expected = None if completed.is_empty() else completed["kickoff_dt"][-1]
        team_row = selected_lags.filter(pl.col("team_id") == team_id)
        observed = team_row["lag1_kickoff_dt"][0]
        if expected is None and observed is not None:
            raise ProspectiveDataError(f"Unexpected current-season lag for {team_id}")
        if expected is not None and observed != expected:
            raise ProspectiveDataError(
                f"Latest completed game is missing from the feature row for {team_id}"
            )


def _game_feature_vector(
    game_id: str,
    snapshot_time: datetime,
    prepared: PreparedFeatures,
) -> tuple[np.ndarray, str]:
    if prepared.as_of >= snapshot_time:
        raise ProspectiveDataError("Feature as-of must predate the odds snapshot")
    selected_lags = prepared.lags.filter(pl.col("game_id") == game_id)
    selected = prepared.materialized.filter(pl.col("game_id") == game_id)
    if selected.height != 2 or selected_lags.height != 2:
        raise ProspectiveDataError("Feature store does not contain exactly two rows for the game")
    validate_lag_boundaries(selected_lags)
    game = prepared.games.filter(pl.col("game_id") == game_id)
    if game.height != 1:
        raise ProspectiveDataError("Canonical game row is missing or duplicated")
    target_kickoff = _parse_utc(game["kickoff_utc"][0], "game kickoff")
    _validate_latest_completed_game(selected_lags, prepared.games, target_kickoff, prepared.as_of)
    home = selected.filter(pl.col("is_home")).select(
        "game_id", *[pl.col(metric).alias(f"home_{metric}") for metric in V11_METRICS]
    )
    away = selected.filter(~pl.col("is_home")).select(
        "game_id", *[pl.col(metric).alias(f"away_{metric}") for metric in V11_METRICS]
    )
    matrix = (
        game.select("game_id", "home_rest", "away_rest")
        .join(home, on="game_id", validate="1:1")
        .join(away, on="game_id", validate="1:1")
        .with_columns(
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_difference"),
            *_matchup_expressions(),
        )
        .select(V11_FEATURE_COLUMNS)
    )
    if matrix.height != 1 or matrix.null_count().row(0) != tuple(0 for _ in matrix.columns):
        raise ProspectiveDataError("Pregame model features are incomplete")
    hash_columns = [
        "game_id",
        "team_id",
        "kickoff_dt",
        *[f"lag{offset}_{metric}" for offset in range(1, 5) for metric in V11_METRICS],
    ]
    row_hash = sha256_bytes(
        selected_lags.select(hash_columns).sort(["game_id", "team_id"]).write_csv().encode("utf-8")
    )
    return matrix.to_numpy(), row_hash


def _anchor_contract(
    rows: list[dict[str, Any]],
    market: str,
    snapshot_time: datetime,
    max_age_minutes: int,
) -> AnchorContract:
    anchor = [row for row in rows if row["bookmaker_key"] == "pinnacle"]
    if len(anchor) != 2:
        return AnchorContract(
            False, "PINNACLE_TWO_SIDED_CONTRACT_MISSING", None, None, None, None, None, None
        )
    orientation = "HOME" if market == "spreads" else "OVER"
    other = "AWAY" if market == "spreads" else "UNDER"
    by_selection = {str(row["selection"]): row for row in anchor}
    if set(by_selection) != {orientation, other}:
        return AnchorContract(
            False, "PINNACLE_SELECTIONS_INVALID", None, None, None, None, None, None
        )
    oriented = by_selection[orientation]
    opposite = by_selection[other]
    lines = [oriented.get("canonical_line"), opposite.get("canonical_line")]
    if lines[0] is None or lines[1] is None:
        return AnchorContract(False, "PINNACLE_POINTS_DIFFER", None, None, None, None, None, None)
    line_value = float(lines[0])
    if not math.isclose(line_value, float(lines[1]), abs_tol=1e-9):
        return AnchorContract(False, "PINNACLE_POINTS_DIFFER", None, None, None, None, None, None)
    overrounds = [float(oriented["overround"]), float(opposite["overround"])]
    if any(value <= 1.0 for value in overrounds):
        return AnchorContract(
            False, "PINNACLE_OVERROUND_INVALID", None, None, None, None, None, None
        )
    updated_values = [
        row.get("source_updated_at_utc") or row.get("last_update_utc") for row in anchor
    ]
    if any(value is None for value in updated_values):
        return AnchorContract(
            False,
            "PINNACLE_UPDATE_TIME_MISSING",
            line_value,
            int(oriented["american_price"]),
            int(opposite["american_price"]),
            float(oriented["vig_free_probability"]),
            float(overrounds[0]),
            None,
        )
    updates = [_parse_utc(value, "Pinnacle update timestamp") for value in updated_values]
    oldest = min(updates)
    if any(update > snapshot_time for update in updates):
        reason = "PINNACLE_UPDATE_AFTER_SNAPSHOT"
        valid = False
    elif snapshot_time - oldest > timedelta(minutes=max_age_minutes):
        reason = "PINNACLE_QUOTE_STALE"
        valid = False
    else:
        reason = None
        valid = True
    return AnchorContract(
        valid,
        reason,
        line_value,
        int(oriented["american_price"]),
        int(opposite["american_price"]),
        float(oriented["vig_free_probability"]),
        float(overrounds[0]),
        max(updates),
    )


def _score_contract(
    candidate: V11Candidate,
    market: str,
    line: float,
    features: np.ndarray,
) -> dict[str, float]:
    if market == "spreads":
        model = candidate.spread_adjustment_model
        weight = candidate.spread_adjustment_weight
        mapper = candidate.spread_residuals
        calibrator = candidate.spread_calibrator
    elif market == "totals":
        model = candidate.total_adjustment_model
        weight = candidate.total_adjustment_weight
        mapper = candidate.total_residuals
        calibrator = candidate.total_calibrator
    else:
        raise ProspectiveDataError(f"Unsupported market: {market}")
    raw_adjustment = float(model.predict(features)[0])
    final_projection = float(line + weight * raw_adjustment)
    empirical = mapper.probabilities(final_projection, line, line)
    non_push_mass = empirical.win + empirical.loss
    if non_push_mass <= 0.0:
        raise ProspectiveDataError("Empirical distribution has no non-push mass")
    raw_non_push = empirical.win / non_push_mass
    calibrated_non_push = float(calibrator.predict(np.asarray([raw_non_push]))[0])
    calibrated_non_push = float(np.clip(calibrated_non_push, 1e-8, 1 - 1e-8))
    model_win = (1.0 - empirical.push) * calibrated_non_push
    model_loss = (1.0 - empirical.push) * (1.0 - calibrated_non_push)
    if not math.isclose(model_win + empirical.push + model_loss, 1.0, abs_tol=1e-9):
        raise ProspectiveDataError("Model outcome probabilities do not sum to one")
    return {
        "raw_adjustment": raw_adjustment,
        "adjustment_weight": float(weight),
        "final_projection": final_projection,
        "raw_non_push_win_probability": float(raw_non_push),
        "calibrated_non_push_win_probability": calibrated_non_push,
        "model_win_probability": float(model_win),
        "model_push_probability": float(empirical.push),
        "model_loss_probability": float(model_loss),
    }


def _prediction_status(
    contract: AnchorContract,
    kickoff: datetime,
    snapshot_time: datetime,
    cutoff: datetime,
    feature_error: str | None,
    policy: dict[str, Any],
) -> str:
    if not contract.valid:
        return f"INELIGIBLE_{contract.reason or 'PINNACLE_CONTRACT'}"
    if kickoff <= cutoff:
        return "INELIGIBLE_BEFORE_PROSPECTIVE_CUTOFF"
    if feature_error:
        return "INELIGIBLE_FEATURES"
    minutes = (kickoff - snapshot_time).total_seconds() / 60.0
    window = cast(dict[str, Any], policy["close_window_minutes_before_kickoff"])
    if minutes > float(window["earliest"]):
        return "DIAGNOSTIC_EARLY_SNAPSHOT"
    if minutes < float(window["latest"]):
        return "INELIGIBLE_TOO_CLOSE_TO_KICKOFF"
    return ELIGIBLE_CLOSE_STATUS


def _equivalent(existing: dict[str, Any], record: dict[str, Any], transient: set[str]) -> bool:
    excluded = {"content_hash", *transient}
    return canonical_hash(existing, excluded=excluded) == canonical_hash(record, excluded=excluded)


def _export_table(name: str, settings: Settings, order_by: str) -> None:
    columns = ARTIFACT_SCHEMAS[name].columns
    with connect(settings) as connection:
        rows = [
            dict(row) for row in connection.execute(f"SELECT * FROM {name} ORDER BY {order_by}")
        ]
    frame = pl.DataFrame(rows).select(columns) if rows else pl.DataFrame({c: [] for c in columns})
    atomic_write_text(settings.root / f"{name}.csv", frame.write_csv())


def _append_ledger(
    name: str,
    records: list[dict[str, Any]],
    settings: Settings,
    *,
    transient: set[str],
    order_by: str,
) -> tuple[int, int]:
    if not records:
        return 0, 0
    schema = ARTIFACT_SCHEMAS[name]
    if len(schema.primary_key) != 1:
        raise AssertionError("Prospective ledgers require a single deterministic primary key")
    key = schema.primary_key[0]
    inserted = 0
    unchanged = 0
    with transaction(settings) as connection:
        placeholders = ",".join("?" for _ in schema.columns)
        for record in records:
            existing_row = connection.execute(
                f"SELECT * FROM {name} WHERE {key}=?", (record[key],)
            ).fetchone()
            if existing_row is not None:
                if not _equivalent(dict(existing_row), record, transient):
                    raise ImmutableLedgerError(f"Immutable {name} conflict for {key}={record[key]}")
                unchanged += 1
                continue
            connection.execute(
                f"INSERT INTO {name} ({','.join(schema.columns)}) VALUES ({placeholders})",
                tuple(record.get(column) for column in schema.columns),
            )
            inserted += 1
    _export_table(name, settings, order_by)
    return inserted, unchanged


def predict_snapshot(
    snapshot_id: str,
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
    *,
    prediction_time: datetime | None = None,
    verify_git: bool = True,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    now = prediction_time or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Prediction time must be timezone-aware")
    now = now.astimezone(UTC)
    bundle = load_frozen_bundle(version, resolved, verify_git=verify_git)
    prepared = _prepare_features(bundle, resolved)
    with connect(resolved) as connection:
        snapshot_row = connection.execute(
            "SELECT * FROM raw_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        request_row = connection.execute(
            "SELECT status FROM api_requests WHERE raw_snapshot_id=?", (snapshot_id,)
        ).fetchone()
        quote_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM market_odds WHERE snapshot_id=? "
                "ORDER BY provider_event_id,market,bookmaker_key,selection",
                (snapshot_id,),
            )
        ]
        consensus_rows = {
            (str(row["provider_event_id"]), str(row["market"])): dict(row)
            for row in connection.execute(
                "SELECT * FROM market_consensus WHERE snapshot_id=?", (snapshot_id,)
            )
        }
        games = {
            str(row["game_id"]): dict(row) for row in connection.execute("SELECT * FROM games")
        }
        existing_ids = {
            str(row["prediction_id"])
            for row in connection.execute(
                "SELECT prediction_id FROM model_predictions "
                "WHERE snapshot_id=? AND model_version=?",
                (snapshot_id, version),
            )
        }
    if snapshot_row is None:
        raise ProspectiveDataError(f"Raw snapshot {snapshot_id} does not exist")
    if request_row is None or request_row["status"] != "COMPLETE":
        raise ProspectiveDataError("Only a completely parsed odds snapshot can be predicted")
    snapshot_time = _parse_utc(snapshot_row["retrieved_at_utc"], "snapshot retrieval time")
    if now < snapshot_time:
        raise ProspectiveDataError("Prediction time cannot precede snapshot retrieval")
    cutoff = _parse_utc(bundle.policy["prospective_start_utc"], "prospective cutoff")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for quote in quote_rows:
        key = (str(quote["provider_event_id"]), str(quote["game_id"]), str(quote["market"]))
        grouped.setdefault(key, []).append(quote)
    records: list[dict[str, Any]] = []
    skipped_after_kickoff = 0
    for (event_id, game_id, market), rows in sorted(grouped.items()):
        game = games.get(game_id)
        if game is None:
            raise ProspectiveDataError(f"Snapshot references unknown game {game_id}")
        kickoff = _parse_utc(game["kickoff_utc"], "game kickoff")
        prediction_id = _stable_id(version, snapshot_id, game_id, market)
        if now >= kickoff:
            if prediction_id not in existing_ids:
                skipped_after_kickoff += 1
            continue
        contract = _anchor_contract(
            rows,
            market,
            snapshot_time,
            int(bundle.policy["quote_max_age_minutes"]),
        )
        feature_error: str | None = None
        features: np.ndarray | None = None
        feature_row_hash: str | None = None
        try:
            features, feature_row_hash = _game_feature_vector(game_id, snapshot_time, prepared)
        except (ProspectiveDataError, ValueError) as exc:
            feature_error = str(exc)
        scores: dict[str, float] = {}
        if contract.valid and contract.line is not None and features is not None:
            scores = _score_contract(bundle.candidate, market, contract.line, features)
        status = _prediction_status(
            contract,
            kickoff,
            snapshot_time,
            cutoff,
            feature_error,
            bundle.policy,
        )
        consensus = consensus_rows.get((event_id, market), {})
        created_at = iso_utc(now)
        pass_reasons = [
            "MODEL_LOCKED_UNTESTED_2026",
            "UNCERTAINTY_UNAVAILABLE_FOR_BETTING",
            status,
        ]
        if feature_error:
            pass_reasons.append(feature_error)
        record: dict[str, Any] = {
            "prediction_id": prediction_id,
            "model_version": version,
            "model_artifact_hash": bundle.artifact_hash,
            "model_spec_hash": bundle.candidate.spec_hash,
            "policy_hash": bundle.policy_hash,
            "git_commit": bundle.git_commit,
            "snapshot_id": snapshot_id,
            "provider_event_id": event_id,
            "game_id": game_id,
            "season": int(game["season"]),
            "week": int(game["week"]),
            "kickoff_utc": iso_utc(kickoff),
            "market": market,
            "orientation": "HOME" if market == "spreads" else "OVER",
            "prediction_created_at_utc": created_at,
            "snapshot_retrieved_at_utc": iso_utc(snapshot_time),
            "feature_as_of_utc": iso_utc(prepared.as_of),
            "feature_input_hash": prepared.input_hash,
            "feature_row_hash": feature_row_hash,
            "pinnacle_updated_at_utc": (
                iso_utc(contract.updated_at) if contract.updated_at is not None else None
            ),
            "pinnacle_line": contract.line,
            "pinnacle_orientation_price": contract.orientation_price,
            "pinnacle_other_price": contract.other_price,
            "pinnacle_orientation_no_vig_probability": contract.orientation_probability,
            "pinnacle_overround": contract.overround,
            "consensus_line": consensus.get("consensus_line"),
            "consensus_probability": consensus.get("consensus_probability"),
            "consensus_status": consensus.get("status", "UNAVAILABLE"),
            "retail_books_count": int(consensus.get("retail_books_count", 0)),
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
            "uncertainty_status": "UNAVAILABLE_FOR_BETTING",
            "eligibility_status": status,
            "decision": "PASS",
            "pass_reason": ";".join(pass_reasons),
            "source": MODEL_SOURCE,
            "retrieved_at_utc": created_at,
            "source_updated_at_utc": (
                iso_utc(contract.updated_at) if contract.updated_at is not None else None
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
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record["eligibility_status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "snapshot_id": snapshot_id,
        "model_version": version,
        "inserted": inserted,
        "unchanged": unchanged,
        "skipped_after_kickoff": skipped_after_kickoff,
        "statuses": statuses,
        "decision": "PASS",
    }


def select_canonical_prediction(
    predictions: list[dict[str, Any]],
    kickoff: datetime,
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    if kickoff.tzinfo is None:
        raise ValueError("Kickoff must be timezone-aware")
    kickoff = kickoff.astimezone(UTC)
    window = cast(dict[str, Any], policy["close_window_minutes_before_kickoff"])
    earliest = float(window["earliest"])
    latest = float(window["latest"])
    max_age = timedelta(minutes=int(policy["quote_max_age_minutes"]))
    valid: list[dict[str, Any]] = []
    for row in predictions:
        if row.get("eligibility_status") != ELIGIBLE_CLOSE_STATUS:
            continue
        snapshot_time = _parse_utc(row["snapshot_retrieved_at_utc"], "snapshot time")
        prediction_time = _parse_utc(row["prediction_created_at_utc"], "prediction time")
        update_time = _parse_utc(row["pinnacle_updated_at_utc"], "Pinnacle update time")
        minutes = (kickoff - snapshot_time).total_seconds() / 60.0
        required = (
            row.get("pinnacle_line"),
            row.get("pinnacle_orientation_no_vig_probability"),
            row.get("calibrated_non_push_win_probability"),
        )
        if (
            latest <= minutes <= earliest
            and prediction_time >= snapshot_time
            and prediction_time < kickoff
            and update_time <= snapshot_time
            and snapshot_time - update_time <= max_age
            and all(value is not None for value in required)
        ):
            valid.append(row)
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            _parse_utc(row["snapshot_retrieved_at_utc"], "snapshot time"),
            str(row["prediction_id"]),
        ),
    )


def _binary_scores(probability: float, outcome: int) -> tuple[float, float]:
    clipped = float(np.clip(probability, 1e-8, 1 - 1e-8))
    brier = (clipped - outcome) ** 2
    log_loss = -(outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped))
    return brier, log_loss


def settle_predictions(
    through: datetime,
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
    *,
    settled_at: datetime | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    if through.tzinfo is None:
        raise ValueError("Settlement cutoff must be timezone-aware")
    through = through.astimezone(UTC)
    now = (settled_at or datetime.now(UTC)).astimezone(UTC)
    policy_path = resolved.manifests_dir / f"prospective_policy_{version}.json"
    policy = cast(dict[str, Any], json.loads(policy_path.read_text(encoding="utf-8")))
    cutoff = _parse_utc(policy["prospective_start_utc"], "prospective cutoff")
    with connect(resolved) as connection:
        games = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM games WHERE season=? AND game_type='REG' AND home_score IS NOT NULL "
                "AND away_score IS NOT NULL ORDER BY week,kickoff_utc,game_id",
                (int(policy["test_season"]),),
            )
        ]
        predictions = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM model_predictions WHERE model_version=? "
                "ORDER BY game_id,market,snapshot_retrieved_at_utc",
                (version,),
            )
        ]
    by_game_market: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for prediction in predictions:
        key = (str(prediction["game_id"]), str(prediction["market"]))
        by_game_market.setdefault(key, []).append(prediction)
    records: list[dict[str, Any]] = []
    for game in games:
        kickoff = _parse_utc(game["kickoff_utc"], "game kickoff")
        if kickoff <= cutoff or kickoff > through:
            continue
        home_margin = float(game["home_score"]) - float(game["away_score"])
        combined_total = float(game["home_score"]) + float(game["away_score"])
        for market, actual, orientation in (
            ("spreads", home_margin, "HOME"),
            ("totals", combined_total, "OVER"),
        ):
            canonical = select_canonical_prediction(
                by_game_market.get((str(game["game_id"]), market), []),
                kickoff,
                policy,
            )
            evaluation_id = _stable_id(version, game["game_id"], market)
            result = "EXCLUDED"
            exclusion_reason: str | None = "NO_VALID_CLOSE_PREDICTION"
            eligible_non_push = 0
            model_brier: float | None = None
            market_brier: float | None = None
            model_log_loss: float | None = None
            market_log_loss: float | None = None
            projection_error: float | None = None
            market_projection_error: float | None = None
            if canonical is not None:
                line = float(canonical["pinnacle_line"])
                projection = float(canonical["final_projection"])
                projection_error = actual - projection
                market_projection_error = actual - line
                if math.isclose(actual, line, abs_tol=1e-9):
                    result = "PUSH"
                    exclusion_reason = "PUSH_EXCLUDED_FROM_PROBABILITY_SCORING"
                else:
                    outcome = int(actual > line)
                    result = "WIN" if outcome else "LOSS"
                    exclusion_reason = None
                    eligible_non_push = 1
                    model_probability = float(canonical["calibrated_non_push_win_probability"])
                    market_probability = float(canonical["pinnacle_orientation_no_vig_probability"])
                    model_brier, model_log_loss = _binary_scores(model_probability, outcome)
                    market_brier, market_log_loss = _binary_scores(market_probability, outcome)
            settled_text = iso_utc(now)
            record: dict[str, Any] = {
                "evaluation_id": evaluation_id,
                "prediction_id": canonical.get("prediction_id") if canonical else None,
                "model_version": version,
                "game_id": game["game_id"],
                "season": int(game["season"]),
                "week": int(game["week"]),
                "kickoff_utc": iso_utc(kickoff),
                "market": market,
                "orientation": orientation,
                "settled_at_utc": settled_text,
                "selection_rule_version": str(policy["selection_rule_version"]),
                "canonical_snapshot_id": canonical.get("snapshot_id") if canonical else None,
                "prediction_created_at_utc": (
                    canonical.get("prediction_created_at_utc") if canonical else None
                ),
                "feature_as_of_utc": canonical.get("feature_as_of_utc") if canonical else None,
                "closing_line": canonical.get("pinnacle_line") if canonical else None,
                "closing_orientation_price": (
                    canonical.get("pinnacle_orientation_price") if canonical else None
                ),
                "closing_other_price": (
                    canonical.get("pinnacle_other_price") if canonical else None
                ),
                "closing_market_probability": (
                    canonical.get("pinnacle_orientation_no_vig_probability") if canonical else None
                ),
                "final_projection": canonical.get("final_projection") if canonical else None,
                "model_non_push_win_probability": (
                    canonical.get("calibrated_non_push_win_probability") if canonical else None
                ),
                "model_win_probability": (
                    canonical.get("model_win_probability") if canonical else None
                ),
                "model_push_probability": (
                    canonical.get("model_push_probability") if canonical else None
                ),
                "model_loss_probability": (
                    canonical.get("model_loss_probability") if canonical else None
                ),
                "actual_value": actual,
                "result": result,
                "eligible_non_push": eligible_non_push,
                "exclusion_reason": exclusion_reason,
                "model_brier": model_brier,
                "market_brier": market_brier,
                "model_log_loss": model_log_loss,
                "market_log_loss": market_log_loss,
                "projection_error": projection_error,
                "market_projection_error": market_projection_error,
                "line_clv": 0.0 if canonical else None,
                "source": "nflverse:results+the-odds-api:pinnacle",
                "retrieved_at_utc": settled_text,
                "source_updated_at_utc": game.get("source_updated_at_utc"),
                "schema_version": resolved.schema_version,
            }
            record["content_hash"] = canonical_hash(record)
            records.append(record)
    inserted, unchanged = _append_ledger(
        "prospective_evaluations",
        records,
        resolved,
        transient={"settled_at_utc", "retrieved_at_utc"},
        order_by="season,week,kickoff_utc,game_id,market,evaluation_id",
    )
    return {
        "model_version": version,
        "through_utc": iso_utc(through),
        "inserted": inserted,
        "unchanged": unchanged,
        "evaluations_considered": len(records),
    }


def _calibration(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    from sklearn.linear_model import LogisticRegression

    if outcomes.size == 0 or np.unique(outcomes).size < 2:
        return {"intercept": None, "slope": None}
    clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs").fit(logits, outcomes)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def _reliability(outcomes: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    if outcomes.size == 0:
        return []
    bins = np.minimum((np.clip(probabilities, 0.0, 1.0) * 10).astype(int), 9)
    rows: list[dict[str, Any]] = []
    for index in range(10):
        mask = bins == index
        if np.any(mask):
            rows.append(
                {
                    "bin": index + 1,
                    "count": int(mask.sum()),
                    "mean_probability": float(np.mean(probabilities[mask])),
                    "observed_rate": float(np.mean(outcomes[mask])),
                }
            )
    return rows


def _prospective_summary(frame: pl.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    minimum = int(policy["minimum_non_push_observations_per_market"])
    markets: dict[str, Any] = {}
    for market in ("spreads", "totals"):
        market_rows = frame.filter(pl.col("market") == market) if frame.height else frame
        eligible = market_rows.filter(pl.col("eligible_non_push") == 1) if frame.height else frame
        outcomes = (
            (eligible["result"] == "WIN").cast(pl.Int64).to_numpy()
            if eligible.height
            else np.asarray([], dtype=int)
        )
        model_probability = (
            eligible["model_non_push_win_probability"].cast(pl.Float64).to_numpy()
            if eligible.height
            else np.asarray([], dtype=float)
        )
        market_probability = (
            eligible["closing_market_probability"].cast(pl.Float64).to_numpy()
            if eligible.height
            else np.asarray([], dtype=float)
        )
        model_brier = (
            float(np.mean((model_probability - outcomes) ** 2)) if eligible.height else None
        )
        market_brier = (
            float(np.mean((market_probability - outcomes) ** 2)) if eligible.height else None
        )
        model_log = (
            float(
                np.mean(
                    -outcomes * np.log(np.clip(model_probability, 1e-8, 1 - 1e-8))
                    - (1 - outcomes) * np.log(np.clip(1 - model_probability, 1e-8, 1 - 1e-8))
                )
            )
            if eligible.height
            else None
        )
        market_log = (
            float(
                np.mean(
                    -outcomes * np.log(np.clip(market_probability, 1e-8, 1 - 1e-8))
                    - (1 - outcomes) * np.log(np.clip(1 - market_probability, 1e-8, 1 - 1e-8))
                )
            )
            if eligible.height
            else None
        )
        projection_rows = (
            market_rows.filter(pl.col("final_projection").is_not_null()) if frame.height else frame
        )
        rmse = (
            float(
                np.sqrt(
                    np.mean(projection_rows["projection_error"].cast(pl.Float64).to_numpy() ** 2)
                )
            )
            if projection_rows.height
            else None
        )
        mae = (
            float(np.mean(np.abs(projection_rows["projection_error"].cast(pl.Float64).to_numpy())))
            if projection_rows.height
            else None
        )
        exclusion_counts = (
            {
                str(row["exclusion_reason"]): int(row["len"])
                for row in market_rows.filter(pl.col("exclusion_reason").is_not_null())
                .group_by("exclusion_reason")
                .len()
                .to_dicts()
            }
            if market_rows.height
            else {}
        )
        line_clv_mean = market_rows["line_clv"].drop_nulls().mean() if market_rows.height else None
        markets[market] = {
            "evaluated_games": market_rows.height,
            "canonical_contracts": int((market_rows["result"] != "EXCLUDED").sum())
            if market_rows.height
            else 0,
            "eligible_non_push": eligible.height,
            "minimum_required": minimum,
            "pushes": int((market_rows["result"] == "PUSH").sum()) if market_rows.height else 0,
            "excluded": int((market_rows["result"] == "EXCLUDED").sum())
            if market_rows.height
            else 0,
            "exclusion_reasons": exclusion_counts,
            "model_brier": model_brier,
            "market_brier": market_brier,
            "model_log_loss": model_log,
            "market_log_loss": market_log,
            "calibration": _calibration(outcomes, model_probability),
            "reliability": _reliability(outcomes, model_probability),
            "projection_rmse": rmse,
            "projection_mae": mae,
            "mean_line_clv": (
                float(cast(float, line_clv_mean)) if line_clv_mean is not None else None
            ),
        }
    ready = all(markets[market]["eligible_non_push"] >= minimum for market in markets)
    coverage = (
        frame.group_by(["week", "market"])
        .agg(
            pl.len().alias("evaluations"),
            pl.col("eligible_non_push").sum().alias("eligible_non_push"),
        )
        .sort(["week", "market"])
        .to_dicts()
        if frame.height
        else []
    )
    return {
        "model_version": policy["model_version"],
        "season": policy["test_season"],
        "prospective_start_utc": policy["prospective_start_utc"],
        "status": "READY_FOR_FORMAL_TEST" if ready else "COLLECTING",
        "decision": "PASS",
        "markets": markets,
        "coverage_by_week": coverage,
    }


def _evaluation_frame(
    version: str, settings: Settings, through_week: int | None = None
) -> pl.DataFrame:
    path = settings.root / "prospective_evaluations.csv"
    frame = pl.read_csv(path, infer_schema_length=100_000)
    if frame.is_empty():
        return frame
    selected = frame.filter(pl.col("model_version") == version)
    if through_week is not None:
        selected = selected.filter(pl.col("week") <= through_week)
    return selected.sort(["season", "week", "kickoff_utc", "game_id", "market"])


def _write_summary_files(
    prefix: Path,
    summary: dict[str, Any],
    audit: pl.DataFrame,
) -> dict[str, str]:
    json_path = Path(f"{prefix}.json")
    html_path = Path(f"{prefix}.html")
    csv_path = Path(f"{prefix}_evaluations.csv")
    reliability_path = Path(f"{prefix}_reliability.csv")
    reliability_rows: list[dict[str, Any]] = []
    for market, values in cast(dict[str, dict[str, Any]], summary["markets"]).items():
        reliability_rows.extend(
            {"market": market, **row} for row in cast(list[dict[str, Any]], values["reliability"])
        )
    atomic_write_text(json_path, json.dumps(summary, sort_keys=True, indent=2))
    atomic_write_text(csv_path, audit.write_csv())
    reliability = (
        pl.DataFrame(reliability_rows)
        if reliability_rows
        else pl.DataFrame(
            schema={
                "market": pl.String,
                "bin": pl.Int64,
                "count": pl.Int64,
                "mean_probability": pl.Float64,
                "observed_rate": pl.Float64,
            }
        )
    )
    atomic_write_text(reliability_path, reliability.write_csv())
    escaped = html.escape(json.dumps(summary, indent=2))
    document = f"""<!doctype html><html><head><meta charset=\"utf-8\">
<title>NFL V1.1 prospective report</title><style>body{{font-family:Arial,sans-serif;
max-width:1100px;margin:32px auto;color:#222}}h1{{color:#17365d}}pre{{background:#f4f7fa;
padding:16px;overflow:auto}}</style></head><body><h1>NFL V1.1 Prospective Report</h1>
<p>Observer mode. Every decision remains <strong>PASS</strong>.</p>
<pre>{escaped}</pre></body></html>"""
    atomic_write_text(html_path, document)
    return {
        "json": str(json_path),
        "html": str(html_path),
        "evaluations_csv": str(csv_path),
        "reliability_csv": str(reliability_path),
    }


def write_prospective_report(
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
    *,
    through_week: int | None = None,
    checkpoint: bool = False,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    policy = cast(
        dict[str, Any],
        json.loads(
            (resolved.manifests_dir / f"prospective_policy_{version}.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    frame = _evaluation_frame(version, resolved, through_week)
    summary = _prospective_summary(frame, policy)
    summary["through_week"] = through_week
    summary["report_type"] = "DESCRIPTIVE_CHECKPOINT" if checkpoint else "CURRENT_STATUS"
    if checkpoint and through_week is None:
        raise ValueError("A checkpoint requires --through-week")
    suffix = f"prospective_checkpoint_week_{through_week}" if checkpoint else "prospective_report"
    paths = _write_summary_files(resolved.reports_dir / f"model_{version}_{suffix}", summary, frame)
    return {**summary, "reports": paths}


def _defer_test(
    version: str,
    policy: dict[str, Any],
    through_week: int,
    status: str,
    spread_rows: int,
    total_rows: int,
    settings: Settings,
) -> dict[str, Any]:
    checked = iso_utc()
    with transaction(settings) as connection:
        existing = connection.execute(
            "SELECT status FROM prospective_test_registry WHERE model_version=? AND test_season=?",
            (version, int(policy["test_season"])),
        ).fetchone()
        if existing is not None and existing["status"] not in {
            "DEFERRED_INSUFFICIENT_SAMPLE",
            "DEFERRED_INCOMPLETE_SETTLEMENT",
            "DEFERRED_WEEK_INCOMPLETE",
        }:
            raise RuntimeError(f"Prospective test registry is already {existing['status']}")
        connection.execute(
            "INSERT INTO prospective_test_registry(model_version,test_season,minimum_week,"
            "requested_through_week,first_checked_at_utc,last_checked_at_utc,status,"
            "spread_non_push_rows,total_non_push_rows) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(model_version,test_season) DO UPDATE SET "
            "requested_through_week=excluded.requested_through_week,"
            "last_checked_at_utc=excluded.last_checked_at_utc,status=excluded.status,"
            "spread_non_push_rows=excluded.spread_non_push_rows,total_non_push_rows=excluded.total_non_push_rows",
            (
                version,
                int(policy["test_season"]),
                int(policy["formal_test_minimum_week"]),
                through_week,
                checked,
                checked,
                status,
                spread_rows,
                total_rows,
            ),
        )
    return {
        "model_version": version,
        "status": status,
        "through_week": through_week,
        "spread_non_push_rows": spread_rows,
        "total_non_push_rows": total_rows,
        "formal_test_consumed": False,
        "decision": "PASS",
    }


def _calibration_gate(values: dict[str, float | None]) -> bool:
    intercept = values["intercept"]
    slope = values["slope"]
    return (
        intercept is not None
        and slope is not None
        and abs(intercept) <= 0.20
        and 0.50 <= slope <= 1.50
    )


def run_prospective_test(
    through_week: int,
    version: str = DEFAULT_MODEL_VERSION,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    initialize_database(resolved)
    policy = cast(
        dict[str, Any],
        json.loads(
            (resolved.manifests_dir / f"prospective_policy_{version}.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    minimum_week = int(policy["formal_test_minimum_week"])
    if through_week < minimum_week:
        raise ValueError(f"Formal prospective testing cannot run before Week {minimum_week}")
    with connect(resolved) as connection:
        prior = connection.execute(
            "SELECT status FROM prospective_test_registry WHERE model_version=? AND test_season=?",
            (version, int(policy["test_season"])),
        ).fetchone()
    if prior is not None and prior["status"] not in {
        "DEFERRED_INSUFFICIENT_SAMPLE",
        "DEFERRED_INCOMPLETE_SETTLEMENT",
        "DEFERRED_WEEK_INCOMPLETE",
    }:
        raise RuntimeError(f"Prospective test has already been consumed: {prior['status']}")

    cutoff = _parse_utc(policy["prospective_start_utc"], "prospective cutoff")
    games = pl.read_csv(resolved.root / "games.csv", infer_schema_length=100_000).with_columns(
        pl.col("season").cast(pl.Int64, strict=False),
        pl.col("week").cast(pl.Int64, strict=False),
        pl.col("kickoff_utc").str.to_datetime(time_zone="UTC", strict=False).alias("kickoff_dt"),
    )
    expected = games.filter(
        (pl.col("season") == int(policy["test_season"]))
        & (pl.col("game_type") == "REG")
        & (pl.col("week") <= through_week)
        & (pl.col("kickoff_dt") > pl.lit(cutoff))
    )
    frame = _evaluation_frame(version, resolved, through_week)
    summary = _prospective_summary(frame, policy)
    spread_rows = int(summary["markets"]["spreads"]["eligible_non_push"])
    total_rows = int(summary["markets"]["totals"]["eligible_non_push"])
    if expected.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null()).height:
        return _defer_test(
            version,
            policy,
            through_week,
            "DEFERRED_WEEK_INCOMPLETE",
            spread_rows,
            total_rows,
            resolved,
        )
    expected_keys = expected.select("game_id").join(
        pl.DataFrame({"market": ["spreads", "totals"]}), how="cross"
    )
    observed_keys = frame.select("game_id", "market") if frame.height else frame
    if expected_keys.join(observed_keys, on=["game_id", "market"], how="anti").height:
        return _defer_test(
            version,
            policy,
            through_week,
            "DEFERRED_INCOMPLETE_SETTLEMENT",
            spread_rows,
            total_rows,
            resolved,
        )
    minimum = int(policy["minimum_non_push_observations_per_market"])
    if spread_rows < minimum or total_rows < minimum:
        checkpoint = write_prospective_report(
            version, resolved, through_week=through_week, checkpoint=True
        )
        deferred = _defer_test(
            version,
            policy,
            through_week,
            "DEFERRED_INSUFFICIENT_SAMPLE",
            spread_rows,
            total_rows,
            resolved,
        )
        deferred["checkpoint_reports"] = checkpoint["reports"]
        return deferred

    # The one-time gate is never consumed from a structurally invalid repository.
    from nfl_bets.validation import validate_all

    validate_all(resolved)
    formal_prefix = resolved.reports_dir / (f"model_{version}_prospective_test_week_{through_week}")
    reserved_paths = [
        Path(f"{formal_prefix}.json"),
        Path(f"{formal_prefix}.html"),
        Path(f"{formal_prefix}_evaluations.csv"),
        Path(f"{formal_prefix}_reliability.csv"),
    ]
    existing_reports = [str(path) for path in reserved_paths if path.exists()]
    if existing_reports:
        raise FileExistsError(
            "Prospective test report paths already exist: " + ", ".join(existing_reports)
        )

    started = iso_utc()
    with transaction(resolved) as connection:
        if prior is None:
            connection.execute(
                "INSERT INTO prospective_test_registry(model_version,test_season,minimum_week,"
                "requested_through_week,first_checked_at_utc,last_checked_at_utc,started_at_utc,"
                "status,spread_non_push_rows,total_non_push_rows) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    version,
                    int(policy["test_season"]),
                    minimum_week,
                    through_week,
                    started,
                    started,
                    started,
                    "IN_PROGRESS",
                    spread_rows,
                    total_rows,
                ),
            )
        else:
            connection.execute(
                "UPDATE prospective_test_registry SET requested_through_week=?,"
                "last_checked_at_utc=?,started_at_utc=?,status='IN_PROGRESS',"
                "spread_non_push_rows=?,total_non_push_rows=? "
                "WHERE model_version=? AND test_season=?",
                (
                    through_week,
                    started,
                    started,
                    spread_rows,
                    total_rows,
                    version,
                    int(policy["test_season"]),
                ),
            )
    try:
        markets = cast(dict[str, dict[str, Any]], summary["markets"])
        gates = {
            "spread_brier_beats_market": markets["spreads"]["model_brier"]
            < markets["spreads"]["market_brier"],
            "spread_log_loss_beats_market": markets["spreads"]["model_log_loss"]
            < markets["spreads"]["market_log_loss"],
            "spread_calibration_acceptable": _calibration_gate(markets["spreads"]["calibration"]),
            "total_brier_beats_market": markets["totals"]["model_brier"]
            < markets["totals"]["market_brier"],
            "total_log_loss_beats_market": markets["totals"]["model_log_loss"]
            < markets["totals"]["market_log_loss"],
            "total_calibration_acceptable": _calibration_gate(markets["totals"]["calibration"]),
        }
        passed = all(gates.values())
        status = (
            "PROSPECTIVE_GATE_PASSED_OBSERVER_ONLY" if passed else "PASS_ONLY_PROSPECTIVE_FAILED"
        )
        summary.update(
            {
                "status": status,
                "through_week": through_week,
                "formal_test_consumed": True,
                "promotion_gates": gates,
                "decision": "PASS",
                "input_hash": sha256_bytes(frame.write_csv().encode("utf-8")),
                "completed_at_utc": iso_utc(),
            }
        )
        paths = _write_summary_files(formal_prefix, summary, frame)
        report_hash = sha256_bytes(Path(paths["json"]).read_bytes())
        completed = iso_utc()
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE prospective_test_registry SET completed_at_utc=?,last_checked_at_utc=?,"
                "status=?,report_path=?,report_hash=? WHERE model_version=? AND test_season=?",
                (
                    completed,
                    completed,
                    status,
                    paths["json"],
                    report_hash,
                    version,
                    int(policy["test_season"]),
                ),
            )
        _append_history(
            {
                "model_version": version,
                "created_at_utc": completed,
                "command": "prospective-test-v11",
                "development_end_season": 2025,
                "test_season": int(policy["test_season"]),
                "spec_hash": policy["spec_hash"],
                "feature_hash": summary["input_hash"],
                "artifact_path": f"artifacts/models/{version}/candidate.joblib",
                "status": status,
                "spread_alpha": policy.get("selected_configuration", {}).get(
                    "spread_adjustment_weight"
                ),
                "total_alpha": policy.get("selected_configuration", {}).get(
                    "total_adjustment_weight"
                ),
                "spread_brier": markets["spreads"]["model_brier"],
                "spread_log_loss": markets["spreads"]["model_log_loss"],
                "spread_market_brier": markets["spreads"]["market_brier"],
                "spread_market_log_loss": markets["spreads"]["market_log_loss"],
                "total_brier": markets["totals"]["model_brier"],
                "total_log_loss": markets["totals"]["model_log_loss"],
                "total_market_brier": markets["totals"]["market_brier"],
                "total_market_log_loss": markets["totals"]["market_log_loss"],
                "notes": "One-time 2026 prospective test; betting remains PASS-only.",
                "source": MODEL_SOURCE,
                "retrieved_at_utc": completed,
                "source_updated_at_utc": None,
                "schema_version": resolved.schema_version,
            },
            resolved,
        )
        _update_project_state(version, status, resolved)
        return {**summary, "reports": paths}
    except BaseException as exc:
        with transaction(resolved) as connection:
            connection.execute(
                "UPDATE prospective_test_registry SET completed_at_utc=?,status='FAILED',"
                "error_message=? WHERE model_version=? AND test_season=?",
                (iso_utc(), str(exc)[:2000], version, int(policy["test_season"])),
            )
        raise


def _update_project_state(version: str, status: str, settings: Settings) -> None:
    path = settings.root / "PROJECT_STATE.md"
    replacements = {
        "- **Last Model Version:**": f"- **Last Model Version:** {version} ({status})",
        "- **System Status:**": (
            f"- **System Status:** V1.1 prospective test status is {status}; "
            "observer controls remain active"
        ),
        "- **Betting Status:**": (
            "- **Betting Status:** PASS-only; uncertainty and staking are not "
            "prospectively validated"
        ),
    }
    lines = path.read_text(encoding="utf-8").splitlines()
    output = [
        next((value for prefix, value in replacements.items() if line.startswith(prefix)), line)
        for line in lines
    ]
    atomic_write_text(path, "\n".join(output) + "\n")
