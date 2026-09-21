from __future__ import annotations

import polars as pl

from nfl_bets.config import Settings
from nfl_bets.db import initialize_database
from nfl_bets.schemas import ARTIFACT_SCHEMAS
from nfl_bets.validation import validate_all


def test_empty_schema_controlled_repository_validates(tmp_path) -> None:
    settings = Settings.for_root(tmp_path)
    initialize_database(settings)
    for name, schema in ARTIFACT_SCHEMAS.items():
        pl.DataFrame({column: [] for column in schema.columns}).write_csv(
            settings.root / f"{name}.csv"
        )
    result = validate_all(settings)
    assert result["status"] == "VALID"
