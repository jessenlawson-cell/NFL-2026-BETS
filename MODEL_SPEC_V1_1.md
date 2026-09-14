# NFL 2026 Betting Model Specification — V1.1

## Status and purpose

V1.1 is a secondary football adjustment to the closing-market spread and total projections. V1.0
failed its untouched 2025 promotion gate and remains permanently `PASS_ONLY`. V1.1 may not produce
BET or LEAN decisions until it earns promotion on post-freeze 2026 prospective observations.

## Temporal contract

- Development uses regular seasons 2014–2025. The 2013 season supplies the initial prior and snap
  continuity history.
- Every rolling signal uses explicit lag 1 through lag 4. A source game's kickoff must be earlier
  than its target game's kickoff.
- Half-life is selected from 1.0, 2.0, and 4.0 inside chronological folds.
- Previous-season prior shrinkage is selected from 0.25, 0.50, 0.75, and 1.00 inside chronological
  folds, with the prior phased out after four current-season games.
- Expanding outer development folds end in 2025. Ridge penalties are selected from 0.1, 1.0,
  10.0, and 100.0 using only earlier seasons within each fold.
- No 2026 outcome, closing line, or post-kick information may enter fitting or hyperparameter
  selection.
- The new untouched population contains only 2026 games whose kickoff is after the immutable
  candidate's `prospective_start_utc` and whose prediction was persisted before kickoff.

## nflverse-derived signals

The V1.1 observation layer retains the eight V1 EPA and success-rate measures and adds:

- Offensive sack rate allowed and defensive sack rate generated per dropback.
- Offensive quarterback-hit rate allowed and defensive quarterback-hit rate generated per
  dropback.
- Offensive and defensive neutral-rush EPA and success rate. Neutral means regular-season,
  non-kneel rushing attempts in quarters 1–3 with an absolute pre-play score differential no
  greater than eight points.
- Offensive and defensive snap continuity. Continuity is the current unit's player snap-share
  overlap with the immediately previous team game, divided by the current unit's total snap-share.

These are nflverse proxies. They must never be labeled PBWR, PRWR, RBWR, or run-stop win rate.

Injury-cluster diagnostics count final Out/Doubtful designations for quarterback, offensive line,
defensive front, and secondary groups, plus Questionable players. The synchronized injury archive
only covers 2025–2026, so these fields are materialized for auditing but disabled as model inputs.

## Matchup transformation and models

For passing, rushing, neutral rushing, pressure risk, and continuity, the system calculates home
and away matchup values and then a directional difference and combined sum. Rest differential is
included separately. This creates 21 standardized secondary features.

Separate ridge models fit the home-margin residual and combined-total residual relative to the
closing market. For each target:

`final_projection = closing_market_projection + beta * ridge_residual_adjustment`

`beta` is estimated from earlier chronological out-of-fold predictions and clipped to `[0, 1]`.
It is never chosen by intuition.

## Probabilities and uncertainty

Empirical residual distributions use out-of-fold projections, spread-depth or total-level terciles,
and `n / (n + 200)` shrinkage toward the global distribution. Spread distributions preserve signed
mass at 3, 6, 7, 10, and 14. Logistic probability calibration uses only stacked chronological
out-of-fold predictions. Development uncertainty uses 1,000 fixed-seed season/game-block bootstrap
samples.

## Prospective promotion boundary

V1.1 stays `LOCKED_UNTESTED_2026` until the prospective sample is complete enough for a frozen
evaluation. A final promotion review requires at least 100 non-push spread observations and 100
non-push total observations after the candidate cutoff. For both markets, calibrated probability
must improve Brier score and log loss over the proportional no-vig closing market. Calibration
intercept must remain within plus or minus 0.20 and slope within 0.50–1.50.

Until every gate passes, all betting decisions remain PASS.
