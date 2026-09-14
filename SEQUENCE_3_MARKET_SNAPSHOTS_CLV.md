# Sequence 3 — Independent Market Snapshots and CLV

## Delivered boundary

Each raw odds board now has one required immutable purpose: `DECISION`, `CLOSE`, or `DIAGNOSTIC`.
The purpose is committed with the raw response and protected by a database trigger. Existing
snapshots migrate conservatively to `DIAGNOSTIC`; no legacy capture is promoted into prospective
evidence. Only DECISION snapshots can create prediction rows. CLOSE snapshots contain market
evidence only.

The existing 16-call weekly pilot is partitioned rather than expanded. Eleven slots capture
DECISION boards and five near-kickoff slots capture CLOSE boards for Thursday night, Sunday early,
Sunday late, Sunday night, and Monday night games. Each scheduled reservation includes its
registered purpose in the request kind and stable idempotency key.

## Independent settlement contract

Settlement chooses the DECISION from the immutable V1.1.2 prediction ledger. It chooses CLOSE
independently from raw and normalized market history without consulting any prediction field. Both
must be valid Pinnacle two-sided contracts in the registered 90-to-5-minute window with quotes no
older than 30 minutes. The close must be a distinct snapshot observed after the decision, and the
records must reconcile by game, provider event, market, orientation, and sportsbook.

If either snapshot is absent, or the records fail identity, freshness, independence, or temporal
checks, `clv_status` explains the failure and every CLV value remains null. Zero is emitted only
when a real calculation from two independent snapshots equals zero.

## CLV measures

- `market_probability_movement`: closing minus decision Pinnacle no-vig probability at each
  snapshot's main line.
- `decision_price_clv`: the closing distribution's conditional no-push probability on the exact
  decision line minus the decision price's break-even probability.
- `decision_line_clv` and compatibility alias `line_clv`: closing canonical line minus decision
  canonical line. Positive is favorable to HOME for spreads and OVER for totals.
- `key_numbers_crossed_json` and `key_number_movement`: signed spread thresholds at 3, 6, 7, 10,
  and 14, distinguishing primary-key crossings or touches from ordinary fractional movement.
- `closing_contract_ev`: `P_close(win) * decision_net_decimal_payout - P_close(loss)`, with push
  return fixed at zero. The frozen empirical residual shape supplies explicit key and push mass and
  is anchored to the independent closing no-vig probability; it is not refit or used to change a
  prediction.

The model and decision-time market scores continue to use the exact frozen DECISION contract.
Missing closing data therefore does not add or remove a V1.1.2 test observation; it only makes CLV
unavailable and prevents the close-dependent diagnostics from being summarized.

## V1.1.2 preservation

Sequence 3 does not edit the model artifact, metadata, specification, prospective policy,
calibrator, probability mapper, feature manifest, prediction formula, prospective cutoff, or
PASS-only rule. It adds observer storage, capture routing, settlement reconciliation, and tests.
The frozen model remains `LOCKED_UNTESTED_2026` and all decisions remain PASS.

## Acceptance record

Verified on `2026-09-14` from the saved local project, based on `observer-app-v1.1.0`:

- full automated suite: 57 tests passed;
- Ruff: passed;
- strict mypy: passed for 23 source files;
- production validation: `VALID` after the observer-schema migration;
- frozen V1.1.2 artifact, metadata, specification, policy, feature manifest, model code, and
  `model_predictions.csv`: unchanged from the Sequence 2 release;
- production prospective rows at migration: zero predictions and zero evaluations;
- legacy raw snapshots: one preserved and conservatively marked `DIAGNOSTIC`;
- provider calls during implementation and verification: zero real calls.
