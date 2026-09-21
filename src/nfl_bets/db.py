from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from nfl_bets.config import Settings, get_settings

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    seasons_json TEXT NOT NULL,
    row_count INTEGER,
    source TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS raw_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    kind TEXT NOT NULL,
    snapshot_purpose TEXT NOT NULL
        CHECK(snapshot_purpose IN ('DECISION','CLOSE','DIAGNOSTIC')),
    path TEXT NOT NULL,
    headers_path TEXT,
    retrieved_at_utc TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    byte_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS api_requests (
    request_id TEXT PRIMARY KEY,
    idempotency_key TEXT,
    attempt_number INTEGER,
    slot TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    week_bucket TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    provider_call_started_at_utc TEXT,
    provider_response_received_at_utc TEXT,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    http_status INTEGER,
    provider_request_id TEXT,
    provider_requests_used INTEGER,
    provider_requests_remaining INTEGER,
    provider_requests_last INTEGER,
    raw_snapshot_id TEXT,
    reconciliation_status TEXT NOT NULL DEFAULT 'NOT_REQUIRED',
    error_message TEXT,
    FOREIGN KEY(raw_snapshot_id) REFERENCES raw_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS capture_reservations (
    idempotency_key TEXT PRIMARY KEY,
    week_bucket TEXT NOT NULL,
    slot TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    active_request_id TEXT,
    provider_call_count INTEGER NOT NULL DEFAULT 0 CHECK(provider_call_count >= 0),
    reservation_conflicts INTEGER NOT NULL DEFAULT 0 CHECK(reservation_conflicts >= 0),
    reconciliation_status TEXT NOT NULL DEFAULT 'NOT_REQUIRED',
    reconciled_at_utc TEXT,
    reconciliation_note TEXT,
    UNIQUE(week_bucket, slot, request_kind),
    FOREIGN KEY(active_request_id) REFERENCES api_requests(request_id)
);

CREATE TABLE IF NOT EXISTS capture_reconciliations (
    reconciliation_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    request_id TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    resolution TEXT NOT NULL,
    note TEXT NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY(idempotency_key) REFERENCES capture_reservations(idempotency_key),
    FOREIGN KEY(request_id) REFERENCES api_requests(request_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY, season INTEGER NOT NULL, week INTEGER NOT NULL,
    game_type TEXT NOT NULL, kickoff_utc TEXT NOT NULL, away_team TEXT NOT NULL,
    home_team TEXT NOT NULL, away_score REAL, home_score REAL, result REAL, total REAL,
    away_rest REAL, home_rest REAL, away_moneyline INTEGER, home_moneyline INTEGER,
    spread_line REAL, away_spread_odds INTEGER, home_spread_odds INTEGER, total_line REAL,
    under_odds INTEGER, over_odds INTEGER, roof TEXT, surface TEXT, temp REAL, wind REAL,
    source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT,
    schema_version TEXT NOT NULL, content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_metrics (
    game_id TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
    kickoff_utc TEXT NOT NULL, team_id TEXT NOT NULL, opponent_team_id TEXT NOT NULL,
    is_home INTEGER NOT NULL, games_available INTEGER NOT NULL, off_pass_epa REAL,
    def_pass_epa REAL, off_rush_epa REAL, def_rush_epa REAL,
    off_pass_success_rate REAL, def_pass_success_rate REAL,
    off_rush_success_rate REAL, def_rush_success_rate REAL, feature_as_of_utc TEXT NOT NULL,
    half_life REAL NOT NULL, source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL,
    source_updated_at_utc TEXT, schema_version TEXT NOT NULL, content_hash TEXT NOT NULL,
    PRIMARY KEY(game_id, team_id), FOREIGN KEY(game_id) REFERENCES games(game_id)
);

CREATE TABLE IF NOT EXISTS market_odds (
    snapshot_id TEXT NOT NULL, provider_event_id TEXT NOT NULL, game_id TEXT,
    commence_time_utc TEXT NOT NULL, bookmaker_key TEXT NOT NULL, bookmaker_title TEXT NOT NULL,
    bookmaker_group TEXT NOT NULL, market TEXT NOT NULL, selection TEXT NOT NULL, point REAL,
    canonical_line REAL, american_price INTEGER NOT NULL, decimal_price REAL NOT NULL,
    implied_probability REAL NOT NULL, vig_free_probability REAL NOT NULL,
    overround REAL NOT NULL, last_update_utc TEXT, source TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT, schema_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, provider_event_id, bookmaker_key, market, selection),
    FOREIGN KEY(snapshot_id) REFERENCES raw_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS injuries (
    season INTEGER NOT NULL, week INTEGER NOT NULL, team_id TEXT NOT NULL,
    player_id TEXT NOT NULL, full_name TEXT, position TEXT, injury_status TEXT,
    practice_status TEXT, report_primary_injury TEXT, report_secondary_injury TEXT,
    snap_share_impact REAL, replacement_quality REAL, source TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT, schema_version TEXT NOT NULL,
    content_hash TEXT NOT NULL, PRIMARY KEY(season, week, team_id, player_id)
);

CREATE TABLE IF NOT EXISTS player_usage (
    season INTEGER NOT NULL, week INTEGER NOT NULL, game_id TEXT NOT NULL,
    player_id TEXT NOT NULL, player_name TEXT, team_id TEXT NOT NULL, position TEXT,
    offense_snaps REAL, offense_pct REAL, defense_snaps REAL, defense_pct REAL,
    special_teams_snaps REAL, special_teams_pct REAL, source TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT, schema_version TEXT NOT NULL,
    content_hash TEXT NOT NULL, PRIMARY KEY(season, week, game_id, player_id, team_id)
);

CREATE TABLE IF NOT EXISTS coverage_metrics (
    season INTEGER NOT NULL, week INTEGER NOT NULL, game_id TEXT NOT NULL,
    team_id TEXT NOT NULL, metric_name TEXT NOT NULL, metric_value REAL,
    availability_status TEXT NOT NULL CHECK(availability_status IN ('UNAVAILABLE','VALIDATED')),
    source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT,
    schema_version TEXT NOT NULL, content_hash TEXT NOT NULL,
    PRIMARY KEY(season, week, game_id, team_id, metric_name)
);

CREATE TABLE IF NOT EXISTS model_history (
    model_version TEXT NOT NULL, created_at_utc TEXT NOT NULL, command TEXT NOT NULL,
    development_end_season INTEGER, test_season INTEGER, spec_hash TEXT, feature_hash TEXT,
    artifact_path TEXT, status TEXT NOT NULL, spread_alpha REAL, total_alpha REAL,
    spread_brier REAL, spread_log_loss REAL, spread_market_brier REAL,
    spread_market_log_loss REAL, total_brier REAL, total_log_loss REAL,
    total_market_brier REAL, total_market_log_loss REAL, notes TEXT, source TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT, schema_version TEXT NOT NULL,
    content_hash TEXT NOT NULL, PRIMARY KEY(model_version, command, created_at_utc)
);

CREATE TABLE IF NOT EXISTS model_test_registry (
    model_version TEXT NOT NULL,
    test_season INTEGER NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    PRIMARY KEY(model_version, test_season)
);

CREATE TABLE IF NOT EXISTS bet_log (
    bet_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, season INTEGER NOT NULL,
    week INTEGER NOT NULL, game_id TEXT NOT NULL, market TEXT NOT NULL, selection TEXT NOT NULL,
    line REAL, price INTEGER, book TEXT, stake REAL NOT NULL, bankroll REAL NOT NULL,
    model_probability REAL, market_fair_probability REAL, expected_roi REAL,
    model_version TEXT NOT NULL, data_timestamp TEXT NOT NULL, closing_line REAL,
    closing_price INTEGER, vig_free_closing_probability REAL, probability_clv REAL,
    result TEXT, profit_loss REAL, decision TEXT NOT NULL, reason TEXT,
    source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT,
    schema_version TEXT NOT NULL, content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_consensus (
    snapshot_id TEXT NOT NULL, game_id TEXT, provider_event_id TEXT NOT NULL,
    market TEXT NOT NULL, anchor_line REAL, retail_line_mean REAL, consensus_line REAL,
    anchor_vig_free_probability REAL, retail_vig_free_probability REAL,
    consensus_probability REAL, same_point INTEGER NOT NULL, retail_books_count INTEGER NOT NULL,
    status TEXT NOT NULL, dislocated_books_json TEXT NOT NULL, source TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT, schema_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, provider_event_id, market)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id TEXT PRIMARY KEY, model_version TEXT NOT NULL,
    model_artifact_hash TEXT NOT NULL, model_spec_hash TEXT NOT NULL,
    policy_hash TEXT NOT NULL, git_commit TEXT NOT NULL, snapshot_id TEXT NOT NULL,
    provider_event_id TEXT NOT NULL, game_id TEXT NOT NULL, season INTEGER NOT NULL,
    week INTEGER NOT NULL, kickoff_utc TEXT NOT NULL, market TEXT NOT NULL,
    orientation TEXT NOT NULL, prediction_created_at_utc TEXT NOT NULL,
    snapshot_retrieved_at_utc TEXT NOT NULL, feature_as_of_utc TEXT,
    feature_input_hash TEXT, feature_row_hash TEXT, pinnacle_updated_at_utc TEXT,
    pinnacle_line REAL, pinnacle_orientation_price INTEGER, pinnacle_other_price INTEGER,
    pinnacle_orientation_no_vig_probability REAL, pinnacle_overround REAL,
    consensus_line REAL, consensus_probability REAL, consensus_status TEXT,
    retail_books_count INTEGER NOT NULL, raw_adjustment REAL, adjustment_weight REAL,
    final_projection REAL, raw_non_push_win_probability REAL,
    calibrated_non_push_win_probability REAL, model_win_probability REAL,
    model_push_probability REAL, model_loss_probability REAL,
    uncertainty_status TEXT NOT NULL, eligibility_status TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision = 'PASS'), pass_reason TEXT NOT NULL,
    source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL, source_updated_at_utc TEXT,
    schema_version TEXT NOT NULL, content_hash TEXT NOT NULL,
    FOREIGN KEY(snapshot_id) REFERENCES raw_snapshots(snapshot_id),
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);

CREATE TABLE IF NOT EXISTS prospective_evaluations (
    evaluation_id TEXT PRIMARY KEY, prediction_id TEXT, model_version TEXT NOT NULL,
    game_id TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
    kickoff_utc TEXT NOT NULL, market TEXT NOT NULL, orientation TEXT NOT NULL,
    settled_at_utc TEXT NOT NULL, selection_rule_version TEXT NOT NULL,
    canonical_snapshot_id TEXT, prediction_created_at_utc TEXT, feature_as_of_utc TEXT,
    decision_snapshot_id TEXT, closing_snapshot_id TEXT,
    decision_line REAL, decision_orientation_price INTEGER, decision_other_price INTEGER,
    decision_market_probability REAL,
    closing_line REAL, closing_orientation_price INTEGER, closing_other_price INTEGER,
    closing_market_probability REAL, final_projection REAL,
    model_non_push_win_probability REAL, model_win_probability REAL,
    model_push_probability REAL, model_loss_probability REAL, actual_value REAL NOT NULL,
    result TEXT NOT NULL, eligible_non_push INTEGER NOT NULL,
    exclusion_reason TEXT, model_brier REAL, market_brier REAL, model_log_loss REAL,
    market_log_loss REAL, projection_error REAL, market_projection_error REAL,
    market_probability_movement REAL, decision_price_clv REAL,
    decision_line_clv REAL, line_clv REAL,
    closing_contract_win_probability REAL, closing_contract_push_probability REAL,
    closing_contract_loss_probability REAL, closing_contract_ev REAL,
    key_numbers_crossed_json TEXT, key_number_movement TEXT,
    clv_status TEXT NOT NULL, reconciliation_status TEXT NOT NULL,
    source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL,
    source_updated_at_utc TEXT, schema_version TEXT NOT NULL, content_hash TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES model_predictions(prediction_id),
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);

CREATE TABLE IF NOT EXISTS prospective_test_registry (
    model_version TEXT NOT NULL, test_season INTEGER NOT NULL,
    minimum_week INTEGER NOT NULL, requested_through_week INTEGER NOT NULL,
    first_checked_at_utc TEXT NOT NULL, last_checked_at_utc TEXT NOT NULL,
    started_at_utc TEXT, completed_at_utc TEXT, status TEXT NOT NULL,
    spread_non_push_rows INTEGER NOT NULL, total_non_push_rows INTEGER NOT NULL,
    report_path TEXT, report_hash TEXT, error_message TEXT,
    PRIMARY KEY(model_version, test_season)
);

CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_market_odds_game ON market_odds(game_id, market);
CREATE INDEX IF NOT EXISTS idx_api_requests_week ON api_requests(week_bucket, started_at_utc);
CREATE INDEX IF NOT EXISTS idx_capture_reservations_week
ON capture_reservations(week_bucket, slot);
CREATE INDEX IF NOT EXISTS idx_capture_reconciliations_key
ON capture_reconciliations(idempotency_key, recorded_at_utc);
CREATE INDEX IF NOT EXISTS idx_predictions_snapshot ON model_predictions(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_predictions_game
ON model_predictions(model_version, game_id, market);
CREATE INDEX IF NOT EXISTS idx_evaluations_week
ON prospective_evaluations(model_version, season, week);
"""


def connect(settings: Settings | None = None) -> sqlite3.Connection:
    resolved = settings or get_settings()
    resolved.db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved.db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


@contextmanager
def transaction(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect(settings)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


@contextmanager
def immediate_transaction(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """Acquire SQLite's write reservation before exposing mutable state to a caller."""
    connection = connect(settings)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
    finally:
        connection.close()


CAPTURE_REQUEST_COLUMNS = {
    "idempotency_key": "TEXT",
    "attempt_number": "INTEGER",
    "provider_call_started_at_utc": "TEXT",
    "provider_response_received_at_utc": "TEXT",
    "provider_request_id": "TEXT",
    "reconciliation_status": "TEXT NOT NULL DEFAULT 'NOT_REQUIRED'",
}

EVALUATION_CLV_COLUMNS = {
    "decision_snapshot_id": "TEXT",
    "closing_snapshot_id": "TEXT",
    "decision_line": "REAL",
    "decision_orientation_price": "INTEGER",
    "decision_other_price": "INTEGER",
    "decision_market_probability": "REAL",
    "market_probability_movement": "REAL",
    "decision_price_clv": "REAL",
    "decision_line_clv": "REAL",
    "closing_contract_win_probability": "REAL",
    "closing_contract_push_probability": "REAL",
    "closing_contract_loss_probability": "REAL",
    "closing_contract_ev": "REAL",
    "key_numbers_crossed_json": "TEXT",
    "key_number_movement": "TEXT",
    "clv_status": "TEXT NOT NULL DEFAULT 'UNAVAILABLE_LEGACY'",
    "reconciliation_status": "TEXT NOT NULL DEFAULT 'LEGACY_UNRECONCILED'",
}


def _migrate_capture_idempotency(connection: sqlite3.Connection) -> None:
    existing = set(table_columns(connection, "api_requests"))
    for name, declaration in CAPTURE_REQUEST_COLUMNS.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE api_requests ADD COLUMN {name} {declaration}")
    legacy_duplicates = connection.execute(
        "SELECT week_bucket,slot,request_kind,COUNT(*) AS count FROM api_requests "
        "WHERE slot!='manual' AND idempotency_key IS NULL "
        "GROUP BY week_bucket,slot,request_kind HAVING COUNT(*)>1"
    ).fetchall()
    if legacy_duplicates:
        raise RuntimeError(
            "Legacy scheduled captures contain duplicate slot keys; reconcile them before migration"
        )
    legacy_attempts = connection.execute(
        "SELECT * FROM api_requests WHERE slot!='manual' AND idempotency_key IS NULL"
    ).fetchall()
    for row in legacy_attempts:
        key = (
            f"scheduled-capture:v1:{row['week_bucket']}:{row['slot']}:{row['request_kind']}"
        )
        if row["status"] == "COMPLETE":
            reservation_status = "COMPLETE"
            reconciliation_status = "NOT_REQUIRED"
        elif row["status"] in {"RAW_SAVED", "HTTP_FAILED", "PARSE_FAILED"}:
            reservation_status = "CLOSED_NO_RETRY"
            reconciliation_status = "NO_RETRY_RAW_SAVED"
        else:
            reservation_status = "RECONCILIATION_REQUIRED"
            reconciliation_status = "LEGACY_PROVIDER_CHECK_REQUIRED"
        connection.execute(
            "UPDATE api_requests SET idempotency_key=?,attempt_number=1,"
            "reconciliation_status=? WHERE request_id=?",
            (key, reconciliation_status, row["request_id"]),
        )
        connection.execute(
            "INSERT INTO capture_reservations("
            "idempotency_key,week_bucket,slot,request_kind,created_at_utc,updated_at_utc,"
            "status,active_request_id,reconciliation_status) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                key,
                row["week_bucket"],
                row["slot"],
                row["request_kind"],
                row["started_at_utc"],
                row["completed_at_utc"] or row["started_at_utc"],
                reservation_status,
                row["request_id"],
                reconciliation_status,
            ),
        )
    connection.execute(
        "UPDATE api_requests SET provider_call_started_at_utc=started_at_utc "
        "WHERE idempotency_key IS NOT NULL AND provider_call_started_at_utc IS NULL "
        "AND (raw_snapshot_id IS NOT NULL OR http_status IS NOT NULL "
        "OR status IN ('AMBIGUOUS_FAILURE','RAW_SAVED','HTTP_FAILED','PARSE_FAILED','COMPLETE'))"
    )
    connection.execute(
        "UPDATE api_requests SET provider_response_received_at_utc=completed_at_utc "
        "WHERE idempotency_key IS NOT NULL AND provider_response_received_at_utc IS NULL "
        "AND completed_at_utc IS NOT NULL AND (raw_snapshot_id IS NOT NULL "
        "OR http_status IS NOT NULL "
        "OR status IN ('RAW_SAVED','HTTP_FAILED','PARSE_FAILED','COMPLETE'))"
    )
    connection.execute(
        "UPDATE capture_reservations SET provider_call_count=("
        "SELECT COUNT(*) FROM api_requests a "
        "WHERE a.idempotency_key=capture_reservations.idempotency_key "
        "AND a.provider_call_started_at_utc IS NOT NULL)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_api_requests_capture_attempt "
        "ON api_requests(idempotency_key, attempt_number) "
        "WHERE idempotency_key IS NOT NULL"
    )
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) "
        "VALUES ('observer-capture-idempotency-1.0.0', "
        "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    )


def _migrate_market_snapshot_clv(connection: sqlite3.Connection) -> None:
    raw_columns = set(table_columns(connection, "raw_snapshots"))
    if "snapshot_purpose" not in raw_columns:
        connection.execute(
            "ALTER TABLE raw_snapshots ADD COLUMN snapshot_purpose TEXT NOT NULL "
            "DEFAULT 'DIAGNOSTIC' CHECK(snapshot_purpose IN "
            "('DECISION','CLOSE','DIAGNOSTIC'))"
        )
    evaluation_columns = set(table_columns(connection, "prospective_evaluations"))
    for name, declaration in EVALUATION_CLV_COLUMNS.items():
        if name not in evaluation_columns:
            connection.execute(
                f"ALTER TABLE prospective_evaluations ADD COLUMN {name} {declaration}"
            )
    connection.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS immutable_raw_snapshot_purpose
        BEFORE UPDATE OF snapshot_purpose ON raw_snapshots
        WHEN NEW.snapshot_purpose != OLD.snapshot_purpose
        BEGIN
            SELECT RAISE(ABORT, 'raw snapshot purpose is immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS immutable_decision_close_snapshot_update
        BEFORE UPDATE ON raw_snapshots
        WHEN OLD.snapshot_purpose IN ('DECISION','CLOSE')
        BEGIN
            SELECT RAISE(ABORT, 'decision and close snapshots are immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS immutable_decision_close_snapshot_delete
        BEFORE DELETE ON raw_snapshots
        WHEN OLD.snapshot_purpose IN ('DECISION','CLOSE')
        BEGIN
            SELECT RAISE(ABORT, 'decision and close snapshots are immutable');
        END;
        CREATE INDEX IF NOT EXISTS idx_raw_snapshots_purpose_time
        ON raw_snapshots(snapshot_purpose, retrieved_at_utc);
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) "
        "VALUES ('observer-market-snapshots-clv-1.0.0', "
        "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    )


def initialize_database(settings: Settings | None = None) -> Path:
    resolved = settings or get_settings()
    with connect(resolved) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) "
            "VALUES ('1.0.0', strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) "
            "VALUES ('1.1.0', strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        )
        _migrate_capture_idempotency(connection)
        _migrate_market_snapshot_clv(connection)
        connection.commit()
    return resolved.db_path


def table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row["name"] for row in connection.execute(f"PRAGMA table_info({table})"))


def database_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).db_path
