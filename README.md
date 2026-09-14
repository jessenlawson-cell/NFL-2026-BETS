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
- `manifests/prospective_policy_1.1.2.json`: immutable live-evaluation contract tied to the
  candidate hash and pre-collection Git tag.
- `src/nfl_bets/`: packaged CLI, ingestion, database, odds, features, and model code.
- `games.csv`, `team_metrics.csv`, `injuries.csv`, `market_odds.csv`,
  `player_usage.csv`, `coverage_metrics.csv`, `model_history.csv`, `bet_log.csv`,
  `model_predictions.csv`, and `prospective_evaluations.csv`:
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
nfl-bets odds snapshot --slot manual --purpose DIAGNOSTIC
nfl-bets v11 predict --snapshot-id <snapshot-id>
nfl-bets settle --through 2026-09-21T12:00:00-04:00
nfl-bets v11 checkpoint --through-week 8
nfl-bets v11 test --through-week 8
nfl-bets report
```

The live odds command requires `THE_ODDS_API_KEY`. It makes one consolidated board request,
persists the raw response before parsing, records quota headers, and never retries an ambiguous
network failure.

Configured scheduled slots are protected by a stable weekly idempotency key. The application
commits a unique SQLite reservation before any provider call and records the call-start boundary,
response receipt, sanitized provider request identifier, and terminal state. Duplicate or
concurrent launches fail before contacting the provider.

Every raw board is immutably labelled `DECISION`, `CLOSE`, or `DIAGNOSTIC`. The registered weekly
pilot partitions its existing 16 calls between decision and near-kickoff close slots. Only
DECISION snapshots may create V1.1.2 predictions. Settlement selects the CLOSE directly from market
history, reports CLV as unavailable if either side is missing, and keeps price movement, line
movement, key-number effects, and push-aware closing-contract EV separate.

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

## Stage 9 observer operation

The model now has an immutable prospective ledger. A prediction uses the exact `1.1.2` artifact,
current leakage-safe V1.1 features, and the Pinnacle point available in a saved odds snapshot.
Every row remains `PASS` because prospective uncertainty and staking have not been validated.

Use `nfl-bets pilot capture --slot <slot-name>` for each of the 16 configured slots during one
complete manual week. This performs exactly one consolidated odds request and immediately writes
the corresponding PASS-only predictions. It does not retry an ambiguous request. Check progress
with:

```powershell
docker compose run --rm nfl-bets pilot preflight --slot monday_1200
docker compose run --rm nfl-bets pilot status --week-bucket 2026-09-15
docker compose run --rm nfl-bets pilot reconcile --slot monday_1200 `
  --week-bucket 2026-09-15
```

`pilot preflight` contacts no provider and spends no credits. It returns `READY` only when the
Docker runtime, local paths, hidden API-key presence, SQLite database, frozen model identity,
schedule, local quota, and duplicate-slot guard all pass. A provider balance is checked when a
previous response has supplied one; otherwise it is reported as a non-blocking warning.

`pilot reconcile` without a resolution is read-only and never contacts the provider. After a
failed attempt, record a resolution only from verified local/provider evidence. The allowed
resolutions are `SAFE_TO_RETRY_NO_CALL`, `PROVIDER_CONFIRMED_NOT_BILLED`, and
`PROVIDER_CONFIRMED_BILLED`; every resolution requires `--note`. No retry is automatic. See
`SEQUENCE_2_CAPTURE_IDEMPOTENCY.md` for the failure-phase rules.

Only after that report says `PASSED` may the Windows tasks be installed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_scheduled_tasks.ps1 `
  -PilotWeekBucket 2026-09-15
```

The installer verifies all 16 manual slots and the Windows Toronto-compatible time zone. Missed
tasks are not replayed. See `STAGE_9_PROSPECTIVE_LEDGER.md` for the operational contract.

## Working rule

Use this repository as the only editable local copy. Keep durable context in tracked files, validate all dataset-derived claims with Python, and commit changes through Git so the workspace remains reproducible.
