from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console

from nfl_bets.data.sync import sync_data
from nfl_bets.db import initialize_database
from nfl_bets.features.build import build_features
from nfl_bets.features.v11 import build_v11_features
from nfl_bets.model.training import test_model, train_model
from nfl_bets.model.v11 import train_v11_model
from nfl_bets.odds.client import snapshot_odds
from nfl_bets.validation import validate_all

app = typer.Typer(
    name="nfl-bets",
    help="Local, reproducible NFL quantitative research and paper-trading CLI.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Synchronize authoritative nflverse data.", no_args_is_help=True)
features_app = typer.Typer(help="Build leakage-safe pregame features.", no_args_is_help=True)
model_app = typer.Typer(help="Train and test frozen model candidates.", no_args_is_help=True)
odds_app = typer.Typer(help="Capture and normalize live market boards.", no_args_is_help=True)
db_app = typer.Typer(help="Initialize local persistence.", no_args_is_help=True)
v11_app = typer.Typer(help="Develop the post-V1 prospective model.", no_args_is_help=True)
v11_features_app = typer.Typer(help="Build V1.1 feature inputs.", no_args_is_help=True)
v11_model_app = typer.Typer(help="Freeze V1.1 candidates.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(model_app, name="model")
app.add_typer(odds_app, name="odds")
app.add_typer(db_app, name="db")
app.add_typer(v11_app, name="v11")
v11_app.add_typer(v11_features_app, name="features")
v11_app.add_typer(v11_model_app, name="model")
console = Console()


def _print_result(result: object) -> None:
    console.print_json(json.dumps(result, default=str))


@db_app.command("init")
def db_init() -> None:
    """Create all authoritative and operational SQLite tables."""
    console.print(f"Initialized {initialize_database()}")


@data_app.command("sync")
def data_sync(
    through: Annotated[int, typer.Option("--through", help="Last season to retrieve.")],
    start_season: Annotated[int, typer.Option("--start-season")] = 2009,
    skip_pbp: Annotated[
        bool,
        typer.Option("--skip-pbp", help="Skip the large play-by-play download for a quick sync."),
    ] = False,
) -> None:
    """Refresh schedules, PBP, team stats, injuries, usage, and manifests via nflreadpy."""
    _print_result(sync_data(through, start_season, include_pbp=not skip_pbp))


@features_app.command("build")
def features_build(
    as_of: Annotated[
        str,
        typer.Option("--as-of", help="Timezone-aware ISO-8601 information cutoff."),
    ],
) -> None:
    """Build explicit lag-1 through lag-4 development feature inputs."""
    parsed_as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    _print_result(build_features(as_of=parsed_as_of))


@model_app.command("train")
def model_train(
    version: Annotated[str, typer.Option("--version", help="Immutable candidate version.")],
) -> None:
    """Run expanding-fold development through 2024 and freeze the candidate."""
    _print_result(train_model(version))


@model_app.command("test")
def model_test(
    version: Annotated[
        str,
        typer.Option("--version", help="Exact immutable candidate version to test."),
    ],
    season: Annotated[int, typer.Option("--season")] = 2025,
) -> None:
    """Consume the untouched 2025 test once and apply the strict promotion gate."""
    _print_result(test_model(version=version, season=season))


@v11_features_app.command("build")
def v11_features_build(
    as_of: Annotated[
        str,
        typer.Option("--as-of", help="Timezone-aware ISO-8601 information cutoff."),
    ],
) -> None:
    """Build leakage-safe V1.1 trench, neutral-rush, and continuity inputs."""
    parsed_as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    _print_result(build_v11_features(as_of=parsed_as_of))


@v11_model_app.command("train")
def v11_model_train(
    version: Annotated[str, typer.Option("--version", help="Immutable 1.1.x version.")],
) -> None:
    """Develop through 2025 and freeze a candidate for post-freeze 2026 evaluation."""
    _print_result(train_v11_model(version))


@odds_app.command("snapshot")
def odds_snapshot(
    slot: Annotated[
        str,
        typer.Option("--slot", help="Configured scheduled slot name, or 'manual'."),
    ],
) -> None:
    """Retrieve one consolidated spreads/totals board without automatic retries."""
    _print_result(snapshot_odds(slot))


@app.command("validate")
def validate() -> None:
    """Run authoritative CSV, SQLite, duplicate-key, join, and leakage-boundary checks."""
    _print_result(validate_all())


if __name__ == "__main__":
    app()
