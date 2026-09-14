from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database
from nfl_bets.odds.client import SCHEDULED_SLOT_PURPOSES, SCHEDULED_SLOTS, _snapshot_purpose
from nfl_bets.pilot import PREFLIGHT_REQUIRED_FILES, pilot_status, preflight_pilot


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
