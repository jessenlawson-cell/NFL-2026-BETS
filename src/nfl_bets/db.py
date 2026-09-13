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
    path TEXT NOT NULL,
    headers_path TEXT,
    retrieved_at_utc TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    byte_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS api_requests (
    request_id TEXT PRIMARY KEY,
    slot TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    week_bucket TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    http_status INTEGER,
    provider_requests_used INTEGER,
    provider_requests_remaining INTEGER,
    provider_requests_last INTEGER,
    raw_snapshot_id TEXT,
    error_message TEXT,
    FOREIGN KEY(raw_snapshot_id) REFERENCES raw_snapshots(snapshot_id)
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

CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_market_odds_game ON market_odds(game_id, market);
CREATE INDEX IF NOT EXISTS idx_api_requests_week ON api_requests(week_bucket, started_at_utc);
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


def initialize_database(settings: Settings | None = None) -> Path:
    resolved = settings or get_settings()
    with connect(resolved) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) "
            "VALUES ('1.0.0', strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        )
    return resolved.db_path


def table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row["name"] for row in connection.execute(f"PRAGMA table_info({table})"))


def database_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).db_path
