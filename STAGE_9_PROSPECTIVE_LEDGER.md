# Stage 9 — Prospective Prediction Ledger

## Delivered state

Model `1.1.2` remains `LOCKED_UNTESTED_2026` and PASS-only. Its candidate binary is bound to the
local annotated tag `v1.1.2-prospective-freeze` and the tracked prospective policy. Every future
prediction records the candidate, policy, feature, and Git identities used to produce it.

The system can now capture an odds board, create a pregame prediction, select a canonical close,
settle completed games, write descriptive reports, and enforce the one-time prospective test.
These steps do not create betting recommendations.

## Canonical evaluation contract

- Pinnacle is the only formal closing benchmark. Retail books and 60/40 consensus remain
  diagnostics.
- A canonical snapshot is the latest valid one from 90 through 5 minutes before kickoff.
- Both sides must be offered at the same point, the overround must exceed 1.00, and the oldest
  Pinnacle quote may be no more than 30 minutes old.
- Prediction creation, snapshot retrieval, and every feature input must predate kickoff.
- The empirical mapper preserves spread push mass. Calibration is applied to the conditional
  non-push probability and then recombined with push mass.
- Settlement selects the prediction using only timing and data-quality fields before loading the
  game result.

## Manual pilot

Store the API key only in the local `.env` file as `THE_ODDS_API_KEY`. Run each configured slot by
hand for one full Tuesday-to-Monday collection week:

```powershell
docker compose run --rm nfl-bets pilot preflight --slot sunday_open_2000
docker compose run --rm nfl-bets pilot capture --slot sunday_open_2000
docker compose run --rm nfl-bets pilot status --week-bucket YYYY-MM-DD
```

Run `pilot preflight` before a capture. It is a local-only, zero-credit check and never displays
the API key. `READY` confirms the container paths, key presence, database, frozen artifact,
schedule, quota guard, and absence of an already completed capture for that slot.

Repeat `pilot capture` using each name in `config/scheduled_slots.toml`. Each capture is one
consolidated spreads/totals request, normally charged as two provider credits. Ambiguous failures
are logged and never automatically retried. If odds capture succeeds but prediction creation
fails, rerun `v11 predict` with the saved snapshot ID instead of spending another credit.

The status report requires all 16 slots, intact raw bodies/headers, parsed quotes, at least one
prediction per snapshot, and zero non-PASS decisions. The scheduler installer is locked until this
report passes.

## Week 8 and the formal gate

After Week 8 outcomes are synchronized, rebuild V1.1 features, settle through the final game, and
create the descriptive checkpoint. Running `v11 test --through-week 8` does not consume the test
when the week is incomplete, settlement is incomplete, or either market has fewer than 100
eligible non-push observations.

Once both markets reach 100, the command consumes the prospective test exactly once. Passing every
Brier, log-loss, and calibration gate records `PROSPECTIVE_GATE_PASSED_OBSERVER_ONLY`; failing any
gate records `PASS_ONLY_PROSPECTIVE_FAILED`. Either result remains PASS-only because betting also
requires a separately frozen uncertainty and staking layer.
