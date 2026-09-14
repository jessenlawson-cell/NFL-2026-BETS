# 2026 NFL Betting Data Dictionary

The executable column contract is `src/nfl_bets/schemas.py`; SQLite mirrors these fields exactly.
Every authoritative artifact includes `source`, `retrieved_at_utc`, `source_updated_at_utc`,
`schema_version`, and a canonical row-level `content_hash`. Blank source-update timestamps mean the
upstream source did not publish one; they never mean retrieval time.

Row hashes exclude `retrieved_at_utc`, so retrieving unchanged business data again produces the
same content hash. Repository-root CSVs are the only authoritative exports. `manifests/` records
run coverage and dataset-level hashes; `data/raw/`, `data/staging/`, and SQLite are local runtime
state rather than competing data authorities.

## `games.csv`

Primary key: `game_id`. Canonical regular-season schedule, result, rest, weather/surface context,
closing spread/total lines, and closing prices. Core identifiers are `season`, `week`, `game_type`,
`kickoff_utc`, `away_team`, and `home_team`. `result` is home score minus away score;
`spread_line` is positive when the home team is favored.

## `team_metrics.csv`

Primary key: (`game_id`, `team_id`). One pregame team snapshot with `kickoff_utc`, opponent,
home/away flag, completed-game count, four-game EWMA offensive/defensive pass/rush EPA and success
rate, `feature_as_of_utc`, and `half_life`. Every current-season rolling input is shifted by one game
before weighting. The ignored lag store preserves lag-1 through lag-4 values and contributing
kickoffs. Model development selects half-life and prior strength using earlier seasons, then writes
one authoritative configuration. These are V1's only football model features besides rest
differential.

## `market_odds.csv`

Append-only primary key: (`snapshot_id`, `provider_event_id`, `bookmaker_key`, `market`,
`selection`). Stores the provider event, nflverse game match, bookmaker group, offered point,
canonical line, American/decimal price, implied and no-vig probability, overround, quote update time,
and retrieval metadata. Spread probability orientation is HOME; total orientation is OVER.

## `injuries.csv`

Primary key: (`season`, `week`, `team_id`, `player_id`). Stores nflreadpy injury and practice report
fields. `snap_share_impact` and `replacement_quality` remain nullable and feature-disabled in V1.

## `player_usage.csv`

Primary key: (`season`, `week`, `game_id`, `player_id`, `team_id`). Stores nflreadpy offense,
defense, and special-teams snap counts and percentages. Validated but feature-disabled in V1.

## `coverage_metrics.csv`

Primary key: (`season`, `week`, `game_id`, `team_id`, `metric_name`). Schema-controlled V2 holding
area. `availability_status` is `UNAVAILABLE` until a validated source exists; values must never be
synthesized.

## `model_history.csv`

Append-only primary key: (`model_version`, `command`, `created_at_utc`). Stores every train/test
event, data/spec hashes, artifact path, market-regression weights, candidate and market Brier/log
loss values, and status. Failed and PASS-only versions are permanent history.

## `bet_log.csv`

Append-only primary key: `bet_id`. Stores the full decision contract, including PASS decisions,
available contract, price, stake/bankroll, model and market probability, expected ROI, version/data
timestamps, close/CLV, result, and profit/loss. Prior predictions and losses are immutable.

## `model_predictions.csv`

Append-only primary key: `prediction_id`, deterministically derived from model version, snapshot,
game, and market. Each row binds the frozen artifact/spec/policy hashes and Git commit to one
pregame Pinnacle contract, feature as-of/hash, ridge adjustment, final projection, calibrated
win/push/loss probabilities, consensus diagnostics, eligibility reason, and mandatory `PASS`
decision. A prediction is never created at or after kickoff and is never updated after insertion.

## `prospective_evaluations.csv`

Append-only primary key: `evaluation_id`, deterministically derived from model version, game, and
market. Settlement first chooses the latest valid Pinnacle prediction captured 90–5 minutes before
kickoff with a quote no older than 30 minutes, then joins the outcome. Rows preserve exclusions and
pushes; pushes never enter Brier or log-loss comparisons. The canonical close row includes model
and market scores, projection errors, and line CLV.

## V1.1 runtime feature store

`data/runtime/features/v11_team_game_lags.parquet` is an ignored, reproducible research artifact.
It stores explicit lag-1 through lag-4 inputs for the V1 signals plus sack rate, quarterback-hit
rate, neutral-rush EPA/success, and offensive/defensive snap continuity. It also carries pregame
injury-cluster diagnostics. The tracked `manifests/v11_feature_inputs.latest.json` records its
definition, source hashes, season coverage, cutoff, and content hash.

Injury diagnostics are not V1.1 model features because historical injury coverage currently begins
in 2025. The authoritative root CSVs remain unchanged; V1.1 does not overwrite V1's frozen
`team_metrics.csv`.

## Operational SQLite tables

`ingestion_runs`, `raw_snapshots`, `api_requests`, `market_consensus`, `model_test_registry`, and
`prospective_test_registry` and `schema_migrations` provide provenance, quota accounting,
consensus diagnostics, one-time test enforcement, and migration history. They are local
infrastructure rather than authoritative CSVs.
