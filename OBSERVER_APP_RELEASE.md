# Observer Application Release Record

## `observer-app-v1.0.0`

- **Application identity:** `observer-app-v1.0.0`, observer-only and `NO_BET`.
- **Observer application baseline commit:** `8609b307338c467424d36336402c42b1434187ba`.
- **Release tag:** local annotated tag `observer-app-v1.0.0`; its peeled commit contains this
  record and the governing review. No branch or tag was pushed by this release task.
- **Model identity used by the application:** V1.1.2, status `LOCKED_UNTESTED_2026`, frozen
  by annotated tag `v1.1.2-prospective-freeze` at
  `44620597db8e2aee1449acbf8128ea222ab8fce5`.
- **Prospective cutoff:** `2026-09-14T01:53:59.778571Z`.
- **Created:** `2026-09-14T16:51:41.0743159Z` by Codex operating directly in the saved
  local project for user `Lawson21`.

## Included application boundary

This release contains the observer application through the Stage 9 prospective ledger: raw-first
odds capture, zero-credit pilot preflight, PASS-only prediction persistence, canonical evaluation
snapshot selection, settlement, descriptive reporting, and the one-time prospective-test guard.
It does not create a challenger, alter model V1.1.2, or authorize a wager.

The application baseline is merge commit `8609b307338c467424d36336402c42b1434187ba`.
Its first parent is the complete observer implementation at `1313bd7a710dc14686918ecc381e6c7a10ba4c71`;
its second parent is local/remote `main` at `3b3d78e463aae3ade21f3a1f2f461a77a6abd3f8`.
The second parent added only ignored `.codex_tmp/` dependencies and ignored generated `outputs/`.
An explicit `ours` merge preserved that commit and its recoverable contents in history without
adding those ignored artifacts to the release tree. No application implementation was discarded.

## Freeze-to-application changes

The following paths changed from frozen model commit `4462059` to observer baseline `8609b30`:

- Documentation/runtime declaration: `DATA_DICTIONARY.md`, `PROJECT_STATE.md`, `README.md`,
  `STAGE_9_PROSPECTIVE_LEDGER.md`, `Dockerfile`.
- Prospective policy and empty authoritative ledgers:
  `manifests/prospective_policy_1.1.2.json`, `model_predictions.csv`,
  `prospective_evaluations.csv`.
- Observer scheduling: `scripts/install_scheduled_tasks.ps1`,
  `scripts/run_scheduled_slot.ps1`.
- Observer application code: `src/nfl_bets/cli.py`, `src/nfl_bets/config.py`,
  `src/nfl_bets/db.py`, `src/nfl_bets/odds/client.py`, `src/nfl_bets/pilot.py`,
  `src/nfl_bets/prospective.py`, `src/nfl_bets/schemas.py`, `src/nfl_bets/validation.py`.
- Observer verification: `tests/integration/test_odds_secret_redaction.py`,
  `tests/integration/test_prospective_prediction.py`,
  `tests/integration/test_prospective_settlement.py`, `tests/unit/test_odds_quota.py`,
  `tests/unit/test_pilot.py`, `tests/unit/test_prospective.py`,
  `tests/unit/test_prospective_gate.py`.

The frozen model specification, V1.1 feature construction/model code, candidate manifest,
development reports, coefficients, feature manifest, candidate artifact, metadata, selected
configuration, probability mappings, and prospective cutoff did not change.

## Frozen identities verified

| Identity | SHA-256 |
| --- | --- |
| Candidate artifact | `6590a4bbb77802175afa4c406de831d596569cbf2e1f596de7df0910f43e8fd5` |
| Candidate metadata | `b6c09bfb3a526520c796a70f8947845a2a23de8f41ae4ebe455225d517ef5ddf` |
| Normalized V1.1 specification | `e51f21289a5f63ada7c1746ad6385412db3f6a4e7fc3ba4e33e43fbdb6e9bf14` |
| Frozen V1.1 feature input | `82bca56d8369d41212515d7d08351e32507304744e010ef5f39ea4798e22d302` |
| V1.1 development feature identity | `70b9b68910f27223b233830d9d0354cce128dce5bf39ef4fcd175d576bc8d64f` |
| Prospective observer policy | `241bade8665d15f1855e8ed5cb5a8f8bc3dbb5e66c87d76b45ab5d76f1caa9f7` |

The repository validator independently returned `VALID` for the candidate and prospective policy,
and every cross-manifest/hash comparison returned true.

## Verification at the application baseline

- `docker compose run --rm --entrypoint python nfl-bets -m pytest` — **47 passed**.
- `docker compose run --rm --entrypoint python nfl-bets -m ruff check .` — **passed**.
- `docker compose run --rm --entrypoint python nfl-bets -m mypy` — **passed in strict mode;
  23 source files checked**.
- `docker compose run --rm nfl-bets validate` — **`VALID`**, including V1.1 freeze and
  prospective-policy verification.
- Frozen tracked-path comparison from `4462059` to `8609b30` — **no differences** in the
  candidate specification, manifest, feature manifest, model/feature code, coefficients, or
  development reports.

## Known limitations and next gate

This remains an observer-only application. It has no independent decision-versus-close market
snapshots, real price/contract CLV, durable scheduled-slot idempotency, provider-side billing
reconciliation, tested backup/restore, bounded log rotation, alerting, clean-clone CI, unbiased
nested mapper/calibrator evaluation, demonstrated predictive edge, validated uncertainty layer,
or staking engine. V1.1.2 must emit only `PASS`; `NO_BET` remains mandatory.

Sequence item 2 may begin only as a separate, explicitly scoped task after this release boundary
passes its final status, diff, tag, and ancestry checks. The next smallest reversible action is to
specify and test a stable scheduled-slot idempotency key and paid-call reconciliation against a
stub provider. No sequence item 2 implementation is included here.
