from __future__ import annotations

import polars as pl

from nfl_bets.teams import canonical_team, canonical_team_column


def test_historical_franchise_aliases_use_current_canonical_ids() -> None:
    assert canonical_team("OAK") == "LV"
    assert canonical_team("SD") == "LAC"
    assert canonical_team("STL") == "LA"
    frame = pl.DataFrame({"team": ["OAK", "SD", "STL", "BUF"]}).with_columns(
        canonical_team_column("team")
    )
    assert frame["team"].to_list() == ["LV", "LAC", "LA", "BUF"]
