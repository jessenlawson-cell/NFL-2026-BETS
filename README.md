# NFL-2026-BETS

The single local workspace for the 2026 NFL quantitative betting project.

## Canonical workspace

Open this repository in both Codex Desktop and VS Code:

`C:\Users\jesse\OneDrive\Documents\GitHub\NFL-2026-BETS`

The GitHub remote is `jessenlawson-cell/NFL-2026-BETS`. Project-specific Codex instructions live in `AGENTS.md`, so they travel with the repository and apply to future Codex tasks started from this folder.

## Current contents

- `AGENTS.md`: operating rules, validation requirements, staking policy, and output contract.
- `PROJECT_STATE.md`: current season, week, model version, and system status.
- `DATA_DICTIONARY.md`: documented dataset fields and keys.
- `MODEL_SPEC.md`: frozen V1 targets, temporal boundaries, calibration, and promotion gate.
- `src/nfl_bets/`: packaged CLI, ingestion, database, odds, features, and model code.
- `games.csv`, `team_metrics.csv`, `injuries.csv`, `market_odds.csv`,
  `player_usage.csv`, `coverage_metrics.csv`, `model_history.csv`, and `bet_log.csv`:
  schema-controlled authoritative exports.
- `tests/fixtures/randomized_team_metrics.csv`: quarantined synthetic data; tests only.
- `data/raw/`, `data/cache/`, and `data/runtime/`: ignored local inputs, cache, and SQLite.
- `.vscode/`: shared Python editor configuration and extension recommendations.
- `NFL-2026-BETS.code-workspace`: one-file entry point for the VS Code workspace.
- `requirements.txt` and `requirements.lock`: direct and fully resolved dependency pins.

## Python setup

The supported runtime is Python 3.13. Docker is the reproducible default:

```powershell
docker compose build
docker compose run --rm nfl-bets db init
```

For a local Python 3.13 installation:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

## Implemented workflow

```powershell
nfl-bets data sync --through 2026
nfl-bets features build --as-of 2026-09-13T12:00:00-04:00
nfl-bets model train --version 1.0.0
nfl-bets model test --version 1.0.0 --season 2025
nfl-bets odds snapshot --slot manual
```

The live odds command requires `THE_ODDS_API_KEY`. It makes one consolidated board request,
persists the raw response before parsing, records quota headers, and never retries an ambiguous
network failure.

## Data and betting warning

The former randomized data generator has been removed and its output quarantined under tests.
Header-only authoritative CSVs are not evidence. The system remains non-actionable until nflverse
data is synchronized, leakage-safe features are built, a candidate is frozen, and the one-time 2025
test promotes it. A failed gate is intentionally released as PASS-only.

## Working rule

Use this repository as the only editable local copy. Keep durable context in tracked files, validate all dataset-derived claims with Python, and commit changes through Git so the workspace remains reproducible.
