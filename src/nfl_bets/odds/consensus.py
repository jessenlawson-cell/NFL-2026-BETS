from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from nfl_bets.util import canonical_hash

ANCHOR_BOOK = "pinnacle"
RETAIL_BOOKS = ("draftkings", "fanduel", "betmgm", "williamhill_us")
BOOK_GROUP = {ANCHOR_BOOK: "anchor", **dict.fromkeys(RETAIL_BOOKS, "retail")}
TEAM_NAME_TO_ID = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


class OddsDataError(ValueError):
    pass


class ZeroJuiceFlatlineError(OddsDataError):
    pass


def american_to_decimal(price: int) -> float:
    if price == 0:
        raise OddsDataError("American odds cannot be zero")
    return 1.0 + (100.0 / abs(price) if price < 0 else price / 100.0)


def implied_probability(price: int) -> float:
    decimal = american_to_decimal(price)
    return 1.0 / decimal


def remove_vig(prices: list[int]) -> tuple[list[float], float]:
    if len(prices) != 2:
        raise OddsDataError("A two-sided market must contain exactly two prices")
    implied = [implied_probability(price) for price in prices]
    overround = sum(implied)
    if overround <= 1.0:
        raise ZeroJuiceFlatlineError(
            f"Zero-juice flatline detected: two-sided overround {overround:.8f} <= 1.00"
        )
    return [probability / overround for probability in implied], overround


@dataclass(frozen=True)
class ParsedBoard:
    quotes: list[dict[str, Any]]
    consensus: list[dict[str, Any]]


def _selection_key(market: str, name: str, home_team_name: str) -> str:
    if market == "spreads":
        return "HOME" if name == home_team_name else "AWAY"
    normalized = name.upper()
    if normalized not in {"OVER", "UNDER"}:
        raise OddsDataError(f"Unexpected totals selection: {name}")
    return normalized


def parse_board(
    payload: list[dict[str, Any]],
    snapshot_id: str,
    retrieved_at_utc: str,
    schema_version: str,
    game_matches: dict[str, str | None],
    freshness_minutes: int = 30,
) -> ParsedBoard:
    quotes: list[dict[str, Any]] = []
    for event in payload:
        event_id = str(event["id"])
        home_name = str(event["home_team"])
        away_name = str(event["away_team"])
        if home_name not in TEAM_NAME_TO_ID or away_name not in TEAM_NAME_TO_ID:
            raise OddsDataError(f"Unmapped provider team in event {event_id}")
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key")
            if book_key not in BOOK_GROUP:
                continue
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                if market_key not in {"spreads", "totals"}:
                    continue
                outcomes = market.get("outcomes", [])
                if len(outcomes) != 2:
                    raise OddsDataError(
                        f"{book_key}/{market_key}/{event_id} is not an exact two-sided market"
                    )
                prices = [int(outcome["price"]) for outcome in outcomes]
                no_vig, overround = remove_vig(prices)
                for outcome, fair_probability in zip(outcomes, no_vig, strict=True):
                    selection = _selection_key(market_key, str(outcome["name"]), home_name)
                    point = float(outcome["point"])
                    canonical_line = (
                        -point if market_key == "spreads" and selection == "HOME" else point
                    )
                    if market_key == "spreads" and selection == "AWAY":
                        canonical_line = point
                    record = {
                        "snapshot_id": snapshot_id,
                        "provider_event_id": event_id,
                        "game_id": game_matches.get(event_id),
                        "commence_time_utc": event["commence_time"],
                        "bookmaker_key": book_key,
                        "bookmaker_title": bookmaker.get("title", book_key),
                        "bookmaker_group": BOOK_GROUP[book_key],
                        "market": market_key,
                        "selection": selection,
                        "point": point,
                        "canonical_line": canonical_line,
                        "american_price": int(outcome["price"]),
                        "decimal_price": american_to_decimal(int(outcome["price"])),
                        "implied_probability": implied_probability(int(outcome["price"])),
                        "vig_free_probability": fair_probability,
                        "overround": overround,
                        "last_update_utc": market.get("last_update", bookmaker.get("last_update")),
                        "source": "the-odds-api",
                        "retrieved_at_utc": retrieved_at_utc,
                        "source_updated_at_utc": market.get(
                            "last_update", bookmaker.get("last_update")
                        ),
                        "schema_version": schema_version,
                    }
                    record["content_hash"] = canonical_hash(record)
                    quotes.append(record)
    return ParsedBoard(
        quotes=quotes,
        consensus=build_consensus(quotes, schema_version, freshness_minutes),
    )


def build_consensus(
    quotes: list[dict[str, Any]],
    schema_version: str,
    freshness_minutes: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for quote in quotes:
        grouped[(quote["provider_event_id"], quote["market"])].append(quote)

    output: list[dict[str, Any]] = []
    for (event_id, market), rows in grouped.items():
        orientation = "HOME" if market == "spreads" else "OVER"
        oriented = [row for row in rows if row["selection"] == orientation]
        if freshness_minutes is not None:
            cutoff = timedelta(minutes=freshness_minutes)

            def is_fresh(row: dict[str, Any], threshold: timedelta = cutoff) -> bool:
                updated_raw = row.get("source_updated_at_utc")
                retrieved_raw = row.get("retrieved_at_utc")
                if not updated_raw or not retrieved_raw:
                    return False
                updated = datetime.fromisoformat(str(updated_raw).replace("Z", "+00:00"))
                retrieved = datetime.fromisoformat(str(retrieved_raw).replace("Z", "+00:00"))
                return timedelta(0) <= retrieved - updated <= threshold

            oriented = [row for row in oriented if is_fresh(row)]
        anchor = next((row for row in oriented if row["bookmaker_key"] == ANCHOR_BOOK), None)
        retail = [row for row in oriented if row["bookmaker_key"] in RETAIL_BOOKS]
        retail_lines = [float(row["canonical_line"]) for row in retail]
        retail_mean = sum(retail_lines) / len(retail_lines) if retail_lines else None
        anchor_line = float(anchor["canonical_line"]) if anchor else None
        if anchor_line is not None and retail_mean is not None:
            consensus_line = 0.60 * anchor_line + 0.40 * retail_mean
        elif anchor_line is not None:
            consensus_line = anchor_line
        else:
            consensus_line = None

        all_lines = ([anchor_line] if anchor_line is not None else []) + retail_lines
        same_point = bool(all_lines) and max(all_lines) - min(all_lines) < 1e-9
        retail_probability = (
            sum(float(row["vig_free_probability"]) for row in retail) / len(retail)
            if retail
            else None
        )
        anchor_probability = float(anchor["vig_free_probability"]) if anchor else None
        consensus_probability = None
        if anchor_probability is not None and retail_probability is not None and same_point:
            consensus_probability = 0.60 * anchor_probability + 0.40 * retail_probability

        if anchor is None:
            status = "RETAIL_ONLY" if retail else "UNAVAILABLE"
        elif not retail:
            status = "NO_RETAIL"
        elif len(retail) < 2:
            status = "DEGRADED_RETAIL"
        elif not same_point:
            status = "INCOMPARABLE_POINTS"
        else:
            status = "VALID"

        dislocated = [
            row["bookmaker_key"]
            for row in retail
            if anchor_line is not None and abs(float(row["canonical_line"]) - anchor_line) >= 0.5
        ]
        template = rows[0]
        record = {
            "snapshot_id": template["snapshot_id"],
            "game_id": template["game_id"],
            "provider_event_id": event_id,
            "market": market,
            "anchor_line": anchor_line,
            "retail_line_mean": retail_mean,
            "consensus_line": consensus_line,
            "anchor_vig_free_probability": anchor_probability,
            "retail_vig_free_probability": retail_probability,
            "consensus_probability": consensus_probability,
            "same_point": int(same_point),
            "retail_books_count": len(retail),
            "status": status,
            "dislocated_books_json": json.dumps(dislocated, separators=(",", ":")),
            "source": "the-odds-api",
            "retrieved_at_utc": template["retrieved_at_utc"],
            "source_updated_at_utc": max(
                (row["source_updated_at_utc"] for row in rows if row["source_updated_at_utc"]),
                default=None,
            ),
            "schema_version": schema_version,
        }
        record["content_hash"] = canonical_hash(record)
        output.append(record)
    return output
