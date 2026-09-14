from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import (
    DuplicateCaptureError,
    capture_reconciliation_status,
    reconcile_scheduled_capture,
    snapshot_odds,
)
from nfl_bets.odds.client import (
    _week_bucket as current_week_bucket,
)

SLOT = "monday_1200"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _week_bucket(settings: Settings) -> str:
    with connect(settings) as connection:
        return str(connection.execute("SELECT week_bucket FROM capture_reservations").fetchone()[0])


def test_concurrent_scheduled_capture_makes_at_most_one_provider_call(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key")
    provider_started = threading.Event()
    release_provider = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        with calls_lock:
            calls += 1
        provider_started.set()
        assert release_provider.wait(timeout=5)
        return httpx.Response(
            200,
            json=[],
            headers={"x-request-id": "provider-1", "x-requests-last": "2"},
        )

    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def capture() -> None:
        client = _client(handler)
        try:
            results.append(snapshot_odds(SLOT, settings=settings, client=client))
        except BaseException as exc:
            errors.append(exc)
        finally:
            client.close()

    first = threading.Thread(target=capture)
    second = threading.Thread(target=capture)
    first.start()
    assert provider_started.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release_provider.set()
    first.join(timeout=5)

    assert calls == 1
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], DuplicateCaptureError)
    assert results[0]["provider_request_id_recorded"] is True
    assert results[0]["snapshot_purpose"] == "DECISION"
    assert "provider_request_id" not in results[0]
    with connect(settings) as connection:
        reservation = connection.execute("SELECT * FROM capture_reservations").fetchone()
        attempts = connection.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0]
        snapshot_purpose = connection.execute(
            "SELECT snapshot_purpose FROM raw_snapshots"
        ).fetchone()[0]
    assert attempts == 1
    assert reservation["status"] == "COMPLETE"
    assert reservation["provider_call_count"] == 1
    assert reservation["reservation_conflicts"] == 1
    assert reservation["request_kind"] == "full-board:decision"
    assert snapshot_purpose == "DECISION"


def test_ambiguous_timeout_blocks_retry_until_provider_confirms_not_billed(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key")
    calls = 0

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("fixture timeout", request=request)

    first_client = _client(timeout_handler)
    with pytest.raises(RuntimeError, match="failed ambiguously"):
        snapshot_odds(SLOT, settings=settings, client=first_client)
    first_client.close()
    bucket = _week_bucket(settings)
    state = capture_reconciliation_status(SLOT, bucket, settings)
    assert state["status"] == "RECONCILIATION_REQUIRED"
    assert state["reconciliation_status"] == "PROVIDER_CHECK_REQUIRED"
    assert state["provider_call_count"] == 1
    assert state["safe_to_call"] is False

    blocked_client = _client(timeout_handler)
    with pytest.raises(DuplicateCaptureError):
        snapshot_odds(SLOT, settings=settings, client=blocked_client)
    blocked_client.close()
    assert calls == 1

    reconciled = reconcile_scheduled_capture(
        SLOT,
        bucket,
        "PROVIDER_CONFIRMED_NOT_BILLED",
        "Stub provider ledger contains no billable request.",
        settings,
    )
    assert reconciled["status"] == "RETRY_AUTHORIZED"
    assert reconciled["safe_to_call"] is True
    assert reconciled["reconciliation_events"] == 1

    def success_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[], headers={"x-request-id": "provider-2"})

    retry_client = _client(success_handler)
    result = snapshot_odds(SLOT, settings=settings, client=retry_client)
    retry_client.close()
    assert calls == 2
    assert result["attempt_number"] == 2
    with connect(settings) as connection:
        attempts = connection.execute(
            "SELECT request_id,attempt_number,status,reconciliation_status FROM api_requests "
            "ORDER BY attempt_number"
        ).fetchall()
        reservation = connection.execute("SELECT * FROM capture_reservations").fetchone()
    assert [row["attempt_number"] for row in attempts] == [1, 2]
    assert attempts[0]["status"] == "RECONCILED_NOT_BILLED"
    assert attempts[1]["status"] == "COMPLETE"
    assert reservation["status"] == "COMPLETE"
    assert reservation["provider_call_count"] == 2
    with connect(settings) as connection:
        event = connection.execute("SELECT * FROM capture_reconciliations").fetchone()
    assert event["resolution"] == "PROVIDER_CONFIRMED_NOT_BILLED"
    assert event["request_id"] == attempts[0]["request_id"]


def test_pre_call_failure_can_only_retry_after_explicit_no_call_reconciliation(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key")
    original_client = httpx.Client

    def fail_client(*args: object, **kwargs: object) -> httpx.Client:
        raise OSError("fixture setup failure")

    monkeypatch.setattr("nfl_bets.odds.client.httpx.Client", fail_client)
    with pytest.raises(RuntimeError, match="before the provider call"):
        snapshot_odds(SLOT, settings=settings)
    bucket = _week_bucket(settings)
    state = capture_reconciliation_status(SLOT, bucket, settings)
    assert state["provider_call_count"] == 0
    assert state["attempt_status"] == "PRE_CALL_FAILED"
    monkeypatch.setattr("nfl_bets.odds.client.httpx.Client", original_client)
    blocked_client = original_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    with pytest.raises(DuplicateCaptureError):
        snapshot_odds(SLOT, settings=settings, client=blocked_client)
    blocked_client.close()

    reconciled = reconcile_scheduled_capture(
        SLOT,
        bucket,
        "SAFE_TO_RETRY_NO_CALL",
        "Local client construction failed before the durable call-start marker.",
        settings,
    )
    assert reconciled["status"] == "RETRY_AUTHORIZED"

    retry_client = _client(lambda _: httpx.Response(200, json=[]))
    result = snapshot_odds(SLOT, settings=settings, client=retry_client)
    retry_client.close()
    assert result["attempt_number"] == 2


def test_response_without_raw_persistence_is_never_automatically_retried(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[], headers={"x-request-id": "provider-raw-failure"})

    monkeypatch.setattr(
        "nfl_bets.odds.client.atomic_write_bytes",
        lambda *_: (_ for _ in ()).throw(OSError("fixture disk failure")),
    )
    client = _client(handler)
    with pytest.raises(RuntimeError, match="raw evidence could not be saved"):
        snapshot_odds(SLOT, settings=settings, client=client)
    client.close()
    bucket = _week_bucket(settings)
    state = capture_reconciliation_status(SLOT, bucket, settings)
    assert state["status"] == "RECONCILIATION_REQUIRED"
    assert state["attempt_status"] == "RAW_PERSIST_FAILED"
    assert state["provider_response_received_at_utc"] is not None
    assert state["provider_request_id_recorded"] is True
    assert "provider_request_id" not in state
    with connect(settings) as connection:
        stored_provider_id = connection.execute(
            "SELECT provider_request_id FROM api_requests"
        ).fetchone()[0]
    assert stored_provider_id == "provider-raw-failure"
    assert state["safe_to_call"] is False

    blocked_client = _client(handler)
    with pytest.raises(DuplicateCaptureError):
        snapshot_odds(SLOT, settings=settings, client=blocked_client)
    blocked_client.close()
    assert calls == 1
    closed = reconcile_scheduled_capture(
        SLOT,
        bucket,
        "PROVIDER_CONFIRMED_BILLED",
        "Provider request provider-raw-failure is present in the stub billing ledger.",
        settings,
    )
    assert closed["status"] == "CLOSED_NO_RETRY"
    assert closed["safe_to_call"] is False


def test_legacy_slot_key_blocks_a_new_purpose_key_without_provider_contact(
    tmp_path, monkeypatch
) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    monkeypatch.setenv("THE_ODDS_API_KEY", "fixture-key")
    with connect(settings) as connection:
        connection.execute(
            "INSERT INTO api_requests(request_id,slot,request_kind,week_bucket,"
            "started_at_utc,completed_at_utc,status) VALUES ('legacy',?,'full-board',?,"
            "'2026-09-14T12:00:00Z','2026-09-14T12:00:01Z','COMPLETE')",
            (SLOT, current_week_bucket(datetime.now(UTC), settings)),
        )
        connection.commit()
    initialize_database(settings)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[])

    client = _client(handler)
    with pytest.raises(DuplicateCaptureError, match="LEGACY_COMPLETE"):
        snapshot_odds(SLOT, settings=settings, client=client)
    client.close()
    assert calls == 0
