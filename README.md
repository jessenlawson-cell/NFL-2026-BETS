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
- `MODEL_SPEC_V1_1.md`: frozen V1.1 secondary-adjustment and prospective-evaluation contract.
- `src/nfl_bets/`: packaged CLI, ingestion, database, odds, features, and model code.
- `games.csv`, `team_metrics.csv`, `injuries.csv`, `market_odds.csv`,
  `player_usage.csv`, `coverage_metrics.csv`, `model_history.csv`, and `bet_log.csv`:
  schema-controlled authoritative exports.
- `tests/fixtures/randomized_team_metrics.csv`: quarantined synthetic data; tests only.
- `data/raw/`, `data/cache/`, `data/staging/`, and `data/runtime/`: ignored provider
  inputs, resumable staging, feature inputs, and SQLite.
- `manifests/`: tracked synchronization, feature, and model provenance records.
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
nfl-bets model train --version 1.0.1
nfl-bets model test --version 1.0.1 --season 2025
nfl-bets v11 features build --as-of 2026-09-13T19:20:14.760705Z
nfl-bets v11 model train --version 1.1.2
nfl-bets odds snapshot --slot manual
```

The live odds command requires `THE_ODDS_API_KEY`. It makes one consolidated board request,
persists the raw response before parsing, records quota headers, and never retries an ambiguous
network failure.

Data synchronization downloads nflverse one season at a time, validates a run in `data/staging/`,
and promotes only complete authoritative exports. `features build` stores explicit lag-1 through
lag-4 inputs; `model train` selects the half-life and prior shrinkage chronologically before writing
the single authoritative `team_metrics.csv` and freezing an untested candidate.

## Data and betting warning

The former randomized data generator has been removed and its output quarantined under tests.
Header-only authoritative CSVs are not evidence. Model `1.0.1` consumed its one-time 2025 test and
was not promoted: spreads missed both market scoring gates, while both spread and total probability
calibration missed the frozen slope limit. The system is therefore permanently PASS-only for V1.
The result is documented in `STAGE_7_2025_TEST.md` and the immutable test report artifacts under
`reports/` and `manifests/`.

V1.1 candidate `1.1.2` is frozen as `LOCKED_UNTESTED_2026`. Its evaluation population begins only
after its recorded prospective cutoff, so earlier 2026 games are ineligible. Injury clusters are
available as diagnostics but model-disabled because the synchronized archive begins in 2025.

## Working rule

Use this repository as the only editable local copy. Keep durable context in tracked files, validate all dataset-derived claims with Python, and commit changes through Git so the workspace remains reproducible.
