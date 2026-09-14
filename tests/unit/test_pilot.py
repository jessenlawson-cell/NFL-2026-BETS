from __future__ import annotations

import json

from nfl_bets.config import Settings
from nfl_bets.db import initialize_database
from nfl_bets.pilot import pilot_status


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
