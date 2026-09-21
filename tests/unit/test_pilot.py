from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import SCHEDULED_SLOT_PURPOSES, SCHEDULED_SLOTS, _snapshot_purpose
from nfl_bets.pilot import (
    PREFLIGHT_REQUIRED_FILES,
    capture_pilot_slot,
    pilot_status,
    preflight_pilot,
)


def _ready_settings(tmp_path) -> Settings:
    settings = Settings.for_root(tmp_path)
    settings.the_odds_api_key = SecretStr("fixture-key")
    settings.ensure_directories()
    for name in PREFLIGHT_REQUIRED_FILES:
        (settings.root / name).write_text("fixture\n", encoding="utf-8")
    slots = "\n".join(f'{slot} = "fixture"' for slot in sorted(SCHEDULED_SLOTS))
    purposes = "\n".join(
        f'{slot} = "{SCHEDULED_SLOT_PURPOSES[slot]}"' for slot in sorted(SCHEDULED_SLOTS)
    )
    config_dir = settings.root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "scheduled_slots.toml").write_text(
        'timezone = "America/Toronto"\n'
        "weekly_call_limit = 20\n"
        "manual_reserve = 4\n\n"
        f"[slots]\n{slots}\n\n[purposes]\n{purposes}\n",
        encoding="utf-8",
    )
    return settings


def test_manual_pilot_stays_incomplete_until_all_sixteen_slots_pass(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    result = pilot_status("2026-09-15", settings=settings)
    assert result["status"] == "INCOMPLETE"
    assert result["required_slots"] == 16
    assert result["passed_slots"] == 0
    report = json.loads(
        (settings.reports_dir / "manual_pilot_2026-09-15.json").read_text(encoding="utf-8")
    )
    assert report["decision_policy"] == "PASS-only"
    assert {row["status"] for row in report["slots"].values()} == {"MISSING"}


def test_scheduled_close_purpose_cannot_be_overridden() -> None:
    assert _snapshot_purpose("sunday_1245", None) == "CLOSE"
    with pytest.raises(ValueError, match="registered as CLOSE"):
        _snapshot_purpose("sunday_1245", "DECISION")


def test_preflight_is_ready_without_contacting_provider(tmp_path, monkeypatch) -> None:
    settings = _ready_settings(tmp_path)
    monkeypatch.setattr("nfl_bets.pilot._running_in_container", lambda: True)
    monkeypatch.setattr("nfl_bets.pilot._runtime_paths_aligned", lambda _: True)
    monkeypatch.setattr(
        "nfl_bets.pilot.load_frozen_bundle",
        lambda **_: SimpleNamespace(policy={"status": "LOCKED_UNTESTED_2026"}),
    )
    result = preflight_pilot(slot="monday_1200", settings=settings)

    assert result["status"] == "READY"
    assert result["provider_contacted"] is False
    assert result["zero_credit_check"] is True
    with connect(settings) as connection:
        assert connection.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0] == 0


def test_preflight_requires_the_frozen_shadow_bundle_when_requested(tmp_path, monkeypatch) -> None:
    settings = _ready_settings(tmp_path)
    monkeypatch.setattr("nfl_bets.pilot._running_in_container", lambda: True)
    monkeypatch.setattr("nfl_bets.pilot._runtime_paths_aligned", lambda _: True)
    monkeypatch.setattr(
        "nfl_bets.pilot.load_frozen_bundle",
        lambda **_: SimpleNamespace(policy={"status": "LOCKED_UNTESTED_2026"}),
    )
    monkeypatch.setattr(
        "nfl_bets.pilot.load_challenger_bundle",
        lambda *_, **__: SimpleNamespace(policy={"status": "SHADOW_FROZEN_WEEK3_TO_8"}),
    )

    result = preflight_pilot(
        slot="monday_1200",
        shadow_version="challenger-0.2.0",
        settings=settings,
    )

    assert result["status"] == "READY"
    assert result["shadow_model_version"] == "challenger-0.2.0"
    assert {row["name"] for row in result["checks"]} >= {"frozen_model", "shadow_model"}


def test_preflight_blocks_missing_key_and_duplicate_slot(tmp_path, monkeypatch) -> None:
    settings = _ready_settings(tmp_path)
    settings.the_odds_api_key = None
    monkeypatch.setattr("nfl_bets.pilot._running_in_container", lambda: True)
    monkeypatch.setattr("nfl_bets.pilot._runtime_paths_aligned", lambda _: True)
    monkeypatch.setattr(
        "nfl_bets.pilot.load_frozen_bundle",
        lambda **_: SimpleNamespace(policy={"status": "LOCKED_UNTESTED_2026"}),
    )
    initialize_database(settings)
    with connect(settings) as connection:
        connection.execute(
            "INSERT INTO api_requests(request_id,slot,request_kind,week_bucket,"
            "started_at_utc,status) VALUES ('fixture','monday_1200','full-board',?,"
            "'2026-09-14T12:00:00Z','COMPLETE')",
            (preflight_pilot(settings=settings)["week_bucket"],),
        )
        connection.commit()

    result = preflight_pilot(slot="monday_1200", settings=settings)

    assert result["status"] == "NOT_READY"
    failures = {row["name"] for row in result["checks"] if row["status"] == "FAIL"}
    assert {"api_key", "duplicate_slot"} <= failures


def test_challenger_failure_never_repeats_provider_capture(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    calls: list[str] = []

    def capture(slot, *, settings, purpose):
        calls.append(slot)
        return {"snapshot_id": "snapshot", "snapshot_purpose": purpose}

    monkeypatch.setattr("nfl_bets.pilot.snapshot_odds", capture)
    monkeypatch.setattr("nfl_bets.pilot.predict_snapshot", lambda *_, **__: {"decision": "PASS"})
    monkeypatch.setattr(
        "nfl_bets.pilot.predict_challenger_snapshot",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )

    result = capture_pilot_slot(
        "wednesday_0900",
        shadow_version="challenger-0.2.0",
        settings=settings,
    )

    assert calls == ["wednesday_0900"]
    assert result["prediction"] == {"decision": "PASS"}
    assert result["shadow_prediction"] == {
        "model_version": "challenger-0.2.0",
        "status": "FAILED",
        "error_type": "RuntimeError",
    }


def test_both_lanes_use_the_same_decision_snapshot(tmp_path, monkeypatch) -> None:
    settings = Settings.for_root(tmp_path)
    monkeypatch.setattr(
        "nfl_bets.pilot.snapshot_odds",
        lambda *_, **__: {"snapshot_id": "shared-snapshot", "snapshot_purpose": "DECISION"},
    )
    monkeypatch.setattr(
        "nfl_bets.pilot.predict_snapshot",
        lambda snapshot_id, **_: {"snapshot_id": snapshot_id, "decision": "PASS"},
    )
    monkeypatch.setattr(
        "nfl_bets.pilot.predict_challenger_snapshot",
        lambda snapshot_id, **_: {"snapshot_id": snapshot_id, "decision": "PASS"},
    )

    result = capture_pilot_slot(
        "wednesday_0900",
        shadow_version="challenger-0.2.0",
        settings=settings,
    )

    assert result["odds"]["snapshot_id"] == "shared-snapshot"
    assert result["prediction"]["snapshot_id"] == "shared-snapshot"
    assert result["shadow_prediction"]["snapshot_id"] == "shared-snapshot"
