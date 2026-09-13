from __future__ import annotations

from nfl_bets.config import Settings
from nfl_bets.db import connect, initialize_database, table_columns
from nfl_bets.schemas import ARTIFACT_SCHEMAS


def test_authoritative_tables_match_csv_contracts(tmp_path) -> None:
    settings = Settings(root=tmp_path)
    initialize_database(settings)
    with connect(settings) as connection:
        for name, schema in ARTIFACT_SCHEMAS.items():
            assert table_columns(connection, name) == schema.columns
