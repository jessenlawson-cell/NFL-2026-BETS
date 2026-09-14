# Stage 8 — V1.1 Development Freeze

## Outcome

Candidate `1.1.2` is frozen as `LOCKED_UNTESTED_2026`. It is not approved for betting.

Development used regular seasons 2014–2025. Only 2026 games kicking off after
`2026-09-14T01:53:59.778571Z`, with predictions saved before kickoff, may enter its future
prospective evaluation.

## New football layer

V1.1 adds nflverse-derived:

- Sack rate allowed and generated per dropback.
- Quarterback-hit rate allowed and generated per dropback.
- Neutral-rush EPA and success rate.
- Offensive and defensive snap continuity.
- Directional and combined matchup transformations.

Neutral rushing is limited to non-kneel rushing attempts in quarters 1–3 when the absolute
pre-play score differential is no greater than eight points. These pressure measures are nflverse
proxies and are not labeled PBWR, PRWR, RBWR, or run-stop win rate.

Injury clusters were materialized for auditing but excluded from the model. The available injury
archive covers only 2025–2026, which is not enough for chronological coefficient validation.

## Data and leakage checks

- V1.1 lag store: 6,842 team-game rows.
- Development matrix: 3,150 games.
- Development coverage: 2014–2025 with zero feature exclusions across every tested configuration.
- Duplicate team-game keys: zero.
- Games without exactly two team rows: zero.
- 2026 rows used in development: zero.
- Every lagged source kickoff predates its target kickoff.

## Selected configuration

| Setting | Value |
|---|---:|
| Four-game half-life | 1.0 |
| Prior strength | 0.25 |
| Spread ridge penalty | 0.1 |
| Total ridge penalty | 100.0 |
| Spread adjustment weight | 0.0 |
| Total adjustment weight | 0.116151 |

The spread adjustment was rejected completely by chronological selection. The total football
adjustment retained only about 11.6% weight.

## Development evidence

The nested-fold spread market RMSE was `12.8104`; market plus the fold-specific adjustment was
`12.8120`. The nested total market RMSE was `13.2558`; market plus adjustment was `13.2638`.

The 1,000-sample bootstrap intervals for adjusted-minus-market RMSE crossed zero for both markets:

| Market | Lower 95% | Median | Upper 95% |
|---|---:|---:|---:|
| Spread | -0.0003 | 0.0010 | 0.0063 |
| Total | -0.0100 | 0.0083 | 0.0290 |

This is not reliable evidence that the secondary point adjustment beats the market. The selected
out-of-fold probability layer scored slightly better than the no-vig market probability in
development and showed acceptable fitted calibration, but that remains development evidence only.

## Integrity correction

Version `1.1.0` was frozen against the V1.0 specification, whose 2025 boundary conflicts with the
V1.1 design. It was preserved and permanently marked `INVALID_SPEC`. Version `1.1.1` was then
found to score ridge and feature choices before applying their chronological adjustment weights;
it was preserved as `INVALID_METHOD`. Nothing was overwritten. Candidate `1.1.2` jointly evaluates
the fitted adjustment weight during selection and is frozen against `MODEL_SPEC_V1_1.md`; its
artifact, metadata, feature, and specification hashes all validate.

## Current decision

V1 remains `PASS_ONLY`. V1.1 remains `LOCKED_UNTESTED_2026`. There have been zero V1.1 test-registry
entries, zero Odds API requests, and zero bet-log entries.
