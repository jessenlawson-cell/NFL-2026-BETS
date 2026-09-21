# Challenger 0.3 Research Sidecar Specification

## Status and isolation

`challenger-0.3.0` is an unweighted, PASS-only research sidecar. It does not alter,
retrain, score, replace, or reinterpret model `1.1.2` or `challenger-0.2.0`. It writes
only under versioned `data/research/challenger_0_3`, `manifests/research/challenger_0_3`,
and `reports/research/challenger_0_3` locations.

Run a capture explicitly with:

```powershell
docker compose run --rm --entrypoint python nfl-bets -m nfl_bets.research_sidecar --season 2026
```

Each capture is immutable and records its source, retrieval time, source-availability
time, schema hash, content hash, and protected operational hashes and row counts. Data
without independently auditable historical publication time is labeled
`RESEARCH_BACKFILL_NOT_OOS_ELIGIBLE` and becomes eligible only for decisions after its
actual capture time.

## Raw research inputs

The sidecar captures schedules, play-by-play, injuries, depth charts, player identifier
crosswalks, snap counts, NGS passing/rushing/receiving, and FTN charting. It uses the
native `nflreadpy.load_nextgen_stats` hook. No research join may use a player name;
`player_gsis_id` is canonical and PFR snap identifiers require a unique captured
GSIS-to-PFR mapping.

NGS, FTN, pace, air-yards, tackle, cornerback, and pass-rusher observations remain raw
or coverage diagnostics. They have no model weight until separately registered and
accepted through chronological ablation.

## QB Personnel Net Leverage candidate

A qualifying observation requires all of the following at the decision cutoff:

- the captured source was available no later than the decision time;
- the latest pre-decision depth snapshot has exactly one GSIS-resolved rank-1 QB and
  one GSIS-resolved rank-2 QB;
- the rank-1 QB has one matching `Out` injury row for the game week;
- the starter has a unique GSIS-to-PFR crosswalk and four prior completed-team-game
  offensive snap-share observations;
- starter and backup have at least one prior qualifying `qb_dropback == 1` PBP row.

Pregame EPA per dropback uses only games before the decision time. Recency half-life,
empirical-prior strength, prior mean, and feature standardization must be fit within the
applicable training fold. The candidate is:

```text
qb_net_le_shock = mean_starter_offense_snap_share_4g
                  * (shrunk_backup_epa_per_dropback
                     - shrunk_starter_epa_per_dropback)
```

Negative values denote a downgrade. No qualifying event is `0.0`; a qualifying event
with missing or ambiguous evidence is null. The only registered game expression is
`home_qb_net_le_shock - away_qb_net_le_shock`. Tackle, cornerback, and pass-rusher
personnel effects remain coverage-only diagnostics.

## Evaluation and acceptance

The unchanged baseline and baseline-plus-candidate must use identical, non-null rows in
nested chronological validation. Every evaluated row must be marked
`PROSPECTIVE_ELIGIBLE`; backfilled rows are rejected. Report Brier score, log loss,
calibration error, and paired bootstrap intervals. Full integration may be considered
only when both Brier and log loss improve out of sample, both paired 95% intervals
exclude zero in the favorable direction, and calibration does not worsen.

Every output remains `RESEARCH_ONLY_UNWEIGHTED` with decision `PASS`. Reports must not
contain a wager, expected ROI, Kelly value, stake, or active risk value.
