from __future__ import annotations

from typing import Any

import polars as pl

TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "LAR": "LA",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "WSH": "WAS",
}


def canonical_team(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return TEAM_ALIASES.get(normalized, normalized)


def canonical_team_column(column: str) -> pl.Expr:
    return pl.col(column).replace(TEAM_ALIASES).alias(column)
