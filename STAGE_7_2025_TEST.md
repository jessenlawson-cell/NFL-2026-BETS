# Stage 7 — Untouched 2025 Test Result

## Decision

Model `1.0.1` is `PASS_ONLY`. It was not promoted for paper betting.

The protected 2025 test was consumed exactly once on 2026-09-13 local time. This result is final
for V1 and must not be used to retune the V1 candidate.

## What was tested

- Frozen candidate: `1.0.1`
- Development cutoff: 2024
- Untouched test season: 2025
- Eligible games: 272
- Closing-market benchmark: proportional no-vig probability at the closing line
- Promotion rule: the candidate had to beat the closing market on Brier score and log loss for
  both spreads and totals, while both calibration slopes remained between 0.50 and 1.50 and both
  calibration intercepts remained within plus or minus 0.20.

## Gate results

| Gate | Result |
|---|---|
| Spread Brier score beats market | FAIL |
| Spread log loss beats market | FAIL |
| Spread calibration acceptable | FAIL |
| Total Brier score beats market | PASS |
| Total log loss beats market | PASS |
| Total calibration acceptable | FAIL |

All six gates were mandatory, so the final decision is `PASS_ONLY`.

## Benchmark scores

| Market | Measure | Candidate | Closing market | Better candidate? |
|---|---:|---:|---:|---|
| Spread | Brier score | 0.249595 | 0.249587 | No |
| Spread | Log loss | 0.692337 | 0.692321 | No |
| Total | Brier score | 0.250283 | 0.250406 | Yes |
| Total | Log loss | 0.693714 | 0.693959 | Yes |

Spread calibration had an intercept of `0.0957` and slope of `2.7358`. Total calibration had an
intercept of `0.1710` and slope of `1.7418`. Both intercepts passed, but both slopes exceeded the
frozen upper limit of `1.50`. The candidate probabilities were too compressed to qualify as
acceptably calibrated.

The raw football projections also had worse RMSE and MAE than the closing market for both targets.
Because development selected market-regression weights of zero, the final point projections were
the market projections; only the residual and calibration layers produced different probabilities.

## Integrity evidence

- Candidate artifact hash remained unchanged during testing.
- Test input hash: `82529d51f9b2c6d4e16d02b2fd6fdb589604677c434f0c2a9930e06896249740`
- The SQLite registry records one completed `PASS_ONLY` evaluation for model `1.0.1`, season 2025.
- Report hashes match the tracked test manifest.
- Odds API requests: zero.
- Bet-log entries: zero.

## Files

- `reports/model_1.0.1_test_2025.html`: readable test report.
- `reports/model_1.0.1_test_2025.json`: exact machine-readable result.
- `reports/model_1.0.1_test_2025_games.csv`: game-level frozen audit output.
- `reports/model_1.0.1_test_2025_gates.csv`: six promotion-gate outcomes.
- `manifests/model_1.0.1_test_2025.json`: input, artifact, and report hashes.
- `artifacts/models/1.0.1/promotion.json`: permanent candidate decision.

## Next stage

V1 must remain PASS-only. V1.1 may use the already specified, nflverse-derived trench, neutral-rush,
snap-continuity, and injury-cluster features. Because 2025 has now informed that work, V1.1 must use
prospective 2026 results as its new untouched evaluation set.
