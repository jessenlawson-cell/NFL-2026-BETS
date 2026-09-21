# NFL 2026 Shadow Challenger Specification

## Status

`challenger-0.2.0` is an experimental, no-stakes comparator. Its official decision is always
`PASS`. It must not alter, retrain, replace, or reinterpret frozen model `1.1.2`.

## Prospective window

The first registered freeze is trained through completed 2026 Week 2. Weeks 3 through 8 are
prospective. Feature definitions, selected columns, coefficients, residual distributions, and
calibration remain frozen during that window. Weekly feature observations may update using only
games completed before the prediction timestamp.

## Inputs

The raw point projections are football-only and do not use a market line as a predictor. Candidate
feature families are derived from nflverse play-by-play and snap counts: passing, rushing,
pressure, neutral PROE and tendency, special teams, continuity, rest, roof, and surface. Injury
clusters are displayed as unweighted warnings. Coverage, numerical injury values, and weather are
disabled until timestamp-complete data exists.

Opponent-adjusted passing and rushing EPA/success features compare each completed-game
observation with the opponent's pregame, lag-only strength. They are optional in selection, never
calculated from the current game's result. Reported power rankings are neutral-field projections
from the selected frozen model against a league-average opponent; they add no separate weights.

## Development

Feature-family combinations, half-life, prior strength, Ridge penalty, model family, blend weight,
and market-regression weight are selected separately for spreads and totals. Candidate model
families are Ridge, deterministic histogram gradient boosting, and fixed Ridge/tree blends. The
market-only forecast is the baseline and a football weight of zero is valid.

Selection uses nested expanding chronological folds through 2025. The outer 2024 and 2025 folds
are untouched by their inner selection. Ranking uses non-push log loss, Brier score, RMSE, feature
count, and model simplicity in that order. Completed Weeks 1 and 2 may enter only the final fit
after every Week 2 game is final.

Empirical residual distributions produce win, push, and loss probabilities; spread distributions
retain signed key-number mass at 3, 6, 7, 10, and 14. Probability calibration is fit only on
chronological out-of-fold predictions. Development evidence records calibration, reliability,
edge buckets, paired bootstrap intervals against market, Ridge/tree disagreement, feature drift,
and complete input/specification hashes.

## Shadow eligibility

Every database decision remains `PASS`. A report may label a selection `SHADOW_CANDIDATE` only
when the DECISION snapshot and paired price pass validation, conservative probability edge is at
least 0.020000, push-aware expected ROI is positive, and no feature or market-integrity failure is
open. Reports contain no Kelly value or stake.

## Evaluation

The challenger and `1.1.2` use identical DECISION and CLOSE snapshots. Week 3-8 comparison uses
matched game-market contracts and reports exclusions rather than imputing missing evidence. The
primary evidence is Brier score, log loss, calibration, projection error, and independent CLV.
Short-run ROI is descriptive only. Week 8 review does not imply promotion.
Outcomes, settlement, CLV, ROI, and comparator results remain sealed until every Week 8 game is
final.
