# NFL 2026 Betting Model Specification — V1.0

## Scope and decision boundary

V1 predicts regular-season home scoring margin and combined total. It is a research and
paper-trading system. The market is both benchmark and shrinkage input. No model result is
eligible for a BET label unless the frozen 2025 test promotes the candidate. Failure of any
promotion gate sets the system to `PASS_ONLY`.

## Temporal contract

- 2009 supplies initial prior history; primary samples are 2010–2025.
- Every team-game feature uses only games with an earlier kickoff.
- Each current-season four-game EWMA is constructed from `shift(1)` through `shift(4)`.
- Half-life is selected from 1.0, 2.0, and 4.0 inside chronological development.
- Weeks 1–4 blend the previous season's final four-game prior with current-season evidence;
  prior weight is zero after four completed current-season games.
- The previous-season team prior is shrunk toward its season league mean using a strength selected
  from 0.25, 0.50, 0.75, and 1.00 inside chronological development.
- Development uses expanding season folds ending in 2024. Hyperparameters, market weights,
  residual mapping, calibration, features, and thresholds are frozen before 2025 evaluation.
- The 2025 test may be consumed once per frozen version. It is never used for tuning V1.

## Features and expected signs

The input contains home and away snapshots for offensive/defensive pass and rush EPA per play
and success rate, plus rest differential. Higher offensive efficiency should raise the relevant
team projection; higher defensive EPA allowed should raise the opponent projection. Ridge signs
are estimated rather than hard-coded and are audited after fitting.

## Models and market regression

Two independent `StandardScaler` plus ridge-regression pipelines predict raw home margin and raw
combined total. Ridge penalty and the separate spread/total market weights are selected using only
walk-forward development predictions. Final projections are
`alpha * raw_projection + (1 - alpha) * closing_market_projection`.

## Probabilities, pushes, and key numbers

Training-only empirical residual distributions convert projections to win/push/loss probability.
Spread mapping preserves explicit signed mass at margins 3, 6, 7, 10, and 14. Spread depth and
total level use training-fold terciles; each stratum is blended with the global distribution using
`n / (n + 200)`. Probabilities are calibrated with a logistic calibration layer fit only to
walk-forward predictions. Pushes are recorded and excluded from binary Brier/log-loss comparisons.

Development uncertainty uses 1,000 fixed-seed season/game-block bootstrap samples. Version 1.0.0
is frozen as `LOCKED_UNTESTED`; training is prohibited from using 2025 outcomes.

## Promotion gate

For both spread and total, the calibrated market-regressed candidate must improve both Brier score
and log loss over the proportional no-vig closing-market probability on untouched 2025 data.
All four comparisons must pass. Calibration intercept must also be within +/-0.20 and calibration
slope within [0.50, 1.50] for each market. Otherwise the version is permanently recorded as
`PASS_ONLY`.
