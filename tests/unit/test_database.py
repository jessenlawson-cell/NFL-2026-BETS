from __future__ import annotations

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database, table_columns
from nfl_bets.schemas import ARTIFACT_SCHEMAS


def test_authoritative_tables_match_csv_contracts(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    with connect(settings) as connection:
        for name, schema in ARTIFACT_SCHEMAS.items():
            assert table_columns(connection, name) == schema.columns


def test_database_backfills_legacy_scheduled_capture_as_reserved_history(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    with connect(settings) as connection:
        connection.execute(
            "INSERT INTO api_requests("
            "request_id,slot,request_kind,week_bucket,started_at_utc,completed_at_utc,status) "
            "VALUES ('legacy','monday_1200','full-board','2026-09-15',"
            "'2026-09-21T16:00:00Z','2026-09-21T16:00:01Z','COMPLETE')"
        )
        connection.commit()

    initialize_database(settings)

    with connect(settings) as connection:
        attempt = connection.execute(
            "SELECT idempotency_key,attempt_number,reconciliation_status "
            "FROM api_requests WHERE request_id='legacy'"
        ).fetchone()
        reservation = connection.execute("SELECT * FROM capture_reservations").fetchone()
    expected = "scheduled-capture:v1:2026-09-15:monday_1200:full-board"
    assert attempt["idempotency_key"] == expected
    assert attempt["attempt_number"] == 1
    assert attempt["reconciliation_status"] == "NOT_REQUIRED"
    assert reservation["idempotency_key"] == expected
    assert reservation["status"] == "COMPLETE"
    assert reservation["provider_call_count"] == 1
