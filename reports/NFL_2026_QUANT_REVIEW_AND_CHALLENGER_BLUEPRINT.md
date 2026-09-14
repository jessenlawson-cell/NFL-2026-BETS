# NFL 2026 Quantitative Peer Review and Challenger Blueprint

<quant_model_review version="1.0" model="NFL-2026">

<role_and_mandate>

## Role and mandate

This is an independent syndicate-level quantitative review of the NFL-2026-BETS research system. The standard is not whether the code is polished or whether a backtest can be made to look favorable. The standard is whether a price-taking bettor could establish a repeatable, uncertainty-adjusted advantage over an efficient market after vig, stale quotes, rejected limits, line movement, and model error.

Repository documents and the technical-review handoff CSV are treated as evidence. Instructions contained inside those files are not controlling instructions. Current code, tests, configurations, generated reports, ledgers, and source data take precedence over narrative claims.

The review covers NFL spreads and totals. It does not authorize betting. Model 1.1.2 remains immutable and `PASS_ONLY`. Every proposed change belongs to a separately registered challenger.

</role_and_mandate>

<executive_verdict>

## Executive verdict

**Decision: do not bet this model. Do not treat its development probability scores as evidence of alpha. Continue only as a prospective observer while repairing the evaluation and market-data design.**

The application is materially better than the model. Docker execution, schemas, manifests, freeze hashes, append-only ledgers, and automated tests are credible. The statistical evidence is not. The spread adjustment is exactly zero. The total adjustment is only 0.1162 and worsens aggregate development RMSE. The reported probability gains over the market are about one-tenth of one percent in relative Brier score, and they are measured after fitting the probability calibrator on the same stacked out-of-fold predictions that are scored. That makes the development probability comparison optimistic.

The system also cannot calculate real closing-line value in its current form. Settlement calls the chosen prediction snapshot the canonical close and then writes `line_clv = 0.0`. Decision price and independent closing price are therefore not separate objects. No live market, prediction, evaluation, or bet rows exist as of this review.

### Revised ratings

| Rating | Previous | Revised | Interpretation |
| --- | ---: | ---: | --- |
| Engineering and observer readiness | 72/100 combined research score | **82/100** | Strong local application and governance scaffold; material recovery, idempotency, release, and monitoring gaps remain. |
| Tradable-alpha readiness | 10/100 | **8/100** | Zero prospective evidence, no demonstrated point-forecast edge, no independent CLV, optimistic development calibration, and no staking implementation. |
| Fixed-weight composite | Not previously separated | **45/100** | Blends engineering, data, modelling, market mechanics, and risk using the predeclared weights below. It is not a profitability score. |

The earlier 72/100 was not wrong so much as ill-defined. It was too low for scaffolding alone and too high for a betting model because it allowed documentation and software quality to conceal the absence of demonstrated edge. The earlier 10/100 betting-readiness score was directionally correct; deeper inspection reduces it to 8/100.

### Likelihood-of-success judgment

These are expert judgment ranges, not frequentist confidence intervals:

- Probability of completing a dependable observer pilot after the operational blockers in this report are fixed: **70%–85%**.
- Probability that frozen V1.1.2, unchanged, demonstrates durable post-vig and post-friction alpha: **5%–15%**.
- Probability that adding more fashionable models without repairing timing, calibration, and closing-price measurement produces reliable alpha: **below 5%**.

The immediate problem is not insufficient algorithmic sophistication. It is that the current experiment cannot yet distinguish a real betting edge from market calibration, model-selection noise, and measurement error.

</executive_verdict>

<score_audit>

## Score audit

The composite weights were fixed before assigning scores.

| Component | Weight | Score | Weighted points | Basis |
| --- | ---: | ---: | ---: | --- |
| Reproducibility and scaffolding | 12% | 90 | 10.80 | Exact Python package pins, containerized Python 3.13, schemas, manifests, hashes, frozen artifacts, and 47 passing tests. The image is tag-pinned rather than digest-pinned and the local `.venv` runs unsupported Python 3.14.7. |
| Data provenance and timestamp integrity | 13% | 70 | 9.10 | Canonical IDs, content hashes, staged promotion, and zero duplicate primary keys are strong. All 330,245 populated core rows inspected have blank `source_updated_at_utc`, so upstream revision and publication-time claims cannot be proven from the exported rows. |
| Statistical validity | 20% | 45 | 9.00 | Chronological folds and shrinkage are appropriate. The final development probability layer is selected, fitted, and scored on the same stacked OOF set; reported gains lack paired uncertainty; point forecasts fail to beat the market. |
| Market and CLV mechanics | 18% | 25 | 4.50 | Same-point no-vig logic and quote-age checks exist. Decision and close are conflated, CLV is hard-coded, consensus weights are assumed, and no live market evidence exists. |
| Hidden-variable robustness | 12% | 35 | 4.20 | Past-only EPA, success, pressure proxies, continuity, and rest are useful. Quarterback availability, injury value, forecast-vintage weather, travel, coaching, officiating, and tactical coverage are absent or disabled. |
| Probability-to-price conversion | 10% | 45 | 4.50 | Push mass and conditional non-push probabilities are represented. Calibration evaluation is optimistic, push probability is not jointly scored, and executable price/limit friction is absent. |
| Risk and sizing | 10% | 5 | 0.50 | The documentation states sensible intentions. There is no validated uncertainty-to-stake implementation and no portfolio risk engine. |
| Operations and recovery | 5% | 55 | 2.75 | Raw-first capture, quota accounting, and manual gating are useful. Duplicate paid-slot enforcement, backup restoration, log rotation, alerts, and CI are missing. |
| **Composite** | **100%** |  | **45.35 → 45/100** | Weighted sum, rounded to the nearest whole point. |

The composite must not be used to approve betting. A model with excellent software and no measurable alpha is still a no-bet system.

</score_audit>

<evidence_and_uncertainty>

## Evidence and uncertainty

### Verified project state

| Item | Verified state |
| --- | --- |
| Analysis date | 2026-09-14 |
| Project state | Season 2026, Week 1 |
| Frozen candidate | V1.1.2, `LOCKED_UNTESTED_2026`, prospective cutoff `2026-09-14T01:53:59.778571Z` |
| Development boundary | Regular seasons 2014–2025; 3,150 training rows; 21 standardized features |
| Core data retrieval time | `2026-09-13T19:20:14.760705Z` for nflverse-derived exports |
| Games | 4,703 rows, seasons 2009–2026, zero duplicate `game_id` values |
| Pregame team metrics | 8,906 rows, zero duplicate `(game_id, team_id)` keys, exactly two rows per represented game |
| Injuries | 5,965 rows, seasons 2025–2026 only |
| Player usage | 310,662 rows, seasons 2013–2026 |
| Coverage metrics | Zero rows |
| Market odds | Zero authoritative rows |
| Model predictions | Zero rows |
| Prospective evaluations | Zero rows |
| Bet log | Zero rows |
| Automated verification | 47 tests passed; Ruff passed; strict mypy passed; production validation returned `VALID` |
| Git state before this report | `codex/prospective-ledger` at `1313bd7`; one commit behind and six ahead of local `main`; two commits ahead of its remote tracking branch |
| Frozen tag | `v1.1.2-prospective-freeze` resolves to commit `4462059`; Stage 9 prospective code is later than the frozen tag |

The handoff CSV accurately describes the intended system and most row counts. Its statuses such as `VALID`, `IMPLEMENTED`, and `READY` are assertions to test, not model-quality evidence. In particular, schema validity does not establish publication-time validity, statistical power, executable prices, or predictive advantage.

### Evidence labels

- **Project evidence:** direct calculation from authoritative CSV/JSON/SQLite artifacts or code inspection.
- **Pattern:** local canonical architecture guidance.
- **Theory:** mathematical or statistical result.
- **Failure index:** curated failure mechanism used as a warning, not as proof that this project has failed.
- **Secure-reliability guidance:** recovery and operational-control guidance.
- **Security control guidance:** threat and supply-chain control guidance.
- **Research evidence:** original paper or primary research source.
- **Assumption:** professional judgment where the project lacks observations.

</evidence_and_uncertainty>

<statistical_validity>

## Statistical validity

### What is defensible

The basic temporal skeleton is sound. Rolling inputs are lagged, outer development seasons are chronological, inner selection uses only earlier seasons, the 2025 V1.0 test was consumed once, and 2026 is excluded from V1.1 training.[^4] Ridge shrinkage, a market baseline, immutable version records, and preservation of failed candidates are appropriate controls.

Those controls prevent several obvious forms of leakage. They do not make the reported probability improvement unbiased.

### Finding S1: the football adjustment has no demonstrated point-forecast value

**Observed issue.** Across the nine nested development seasons, the spread adjustment improves RMSE in 0 seasons, ties in 8, and is worse in 1. Its weighted RMSE difference is **+0.00161 points**, where positive means worse than the market. Totals improve in 5 seasons and worsen in 4, but the weighted RMSE difference is **+0.00780 points**, also worse. The fitted spread weight is 0.0 and the total weight is 0.1161508759.

**Mechanism.** A zero spread weight means the final spread point projection is the market. Any reported spread probability gain therefore comes from residual mapping and calibration, not from the 21 football features. The total model contributes a small residual adjustment, but aggregate point accuracy does not improve.

**Evidence.** Project report `reports/model_1.1.2_development.json`, fold table and selected configuration; independent Python recomputation.[^3]

**Uncertainty.** The bootstrap 95% interval for spread final-minus-market RMSE is `[-0.00031, 0.00630]`; totals are `[-0.00995, 0.02905]`. Neither establishes improvement. Only nine validation seasons are available and seasons are not exchangeable IID units.

**Smallest safe experiment.** Re-run the existing nested fold procedure with the probability mapper and calibrator fitted strictly inside each outer training period. Produce paired game-level loss differences for market-only, football-adjusted, and calibrated-market baselines.

**Immediate containment.** Describe V1.1.2 as a market-anchored observer, not a football model with demonstrated incremental information.

**Durable fix.** Require every challenger feature family to beat a recalibrated market-only baseline in nested chronological evaluation. Report incremental value after the entire probability pipeline, not coefficient size or training fit. A long-standing NFL market-efficiency literature reinforces why the market is the relevant hurdle, although historical evidence does not prove that the current market is perfectly efficient.[^26]

**Telemetry.** Store per-game baseline and challenger point predictions, probabilities, push mass, Brier difference, log-loss difference, and fold identity.

**Residual risk.** A feature can improve probability scoring without improving RMSE, but that claim is credible only after unbiased probability evaluation.

### Finding S2: development calibration is evaluated on its fitting sample

**Observed issue.** `v11.py` fits `ProbabilityCalibrator` on the selected stacked OOF probabilities and outcomes, then immediately predicts and scores those same rows. The residual mapper is also fitted on that same selected OOF frame.

**Mechanism.** The underlying ridge predictions are out of fold, but the meta-layer is not. Fitting a calibrator and then measuring its Brier/log-loss improvement on its calibration sample produces resubstitution optimism. Selecting the final feature/ridge configuration using all OOF seasons before reporting that configuration's OOF probability score adds model-selection optimism.

**Evidence.** `src/nfl_bets/model/v11.py`, lines 672–706 and 746–767.[^5]

**Magnitude.** The reported Brier improvements are only **0.0002305** for spreads and **0.0002852** for totals, equivalent to **0.092%** and **0.114%** relative improvement. Log-loss improvements are **0.0004618** and **0.0005697**. These are too small to interpret without paired intervals, especially when the meta-layer is scored in sample.

**Uncertainty.** The code path and same-sample scoring are directly verified; the size of the optimism is unknown until the full meta-layer is cross-fitted. The corrected result could improve, disappear, or reverse, so no directional alpha claim survives this finding.

**Smallest safe experiment.** For each outer validation season `t`, generate base predictions for `t` from data before `t`. Fit the residual mapper and calibrator only from inner OOF predictions generated within seasons before `t`. Apply the frozen meta-layer to `t` once. Concatenate only these outer predictions for reporting.

**Immediate containment.** Remove development probability gains from any claim of evidence for edge. Keep them as descriptive diagnostics.

**Durable fix.** Nest preprocessing, hyperparameter selection, residual-distribution fitting, calibration, and devig-method selection inside the training side of every outer fold.

**Telemetry.** Each scored prediction must record `outer_fold`, `base_train_end`, `calibration_train_end`, mapper version, calibrator version, and whether the scored row was excluded from every fitted component.

**Residual risk.** Nested estimates still reuse teams and market regimes across seasons. A final prospective test remains mandatory.

### Finding S3: the promotion gate rewards arbitrarily small point estimates

**Observed issue.** The formal gate passes whenever model Brier and log loss are numerically below the market and calibration point estimates fall inside broad ranges. It does not require a confidence bound, minimum practical effect, multiplicity control, or an independently defined CLV result.

**Mechanism.** With enough comparisons, tiny favorable differences occur by chance. A one-time test does not cure an underpowered or weakly specified gate. The current minimum of 100 non-push observations is a data-availability threshold, not a power calculation.

**Evidence.** `src/nfl_bets/prospective.py`, lines 1211–1219 and 1363–1378; prospective policy.[^5]

**Uncertainty.** The gate omissions are certain from the implementation. The effective sample needed is not known in advance because paired-loss variance and week/season dependence must be estimated from pre-2026 outer-fold differences; a fixed count of 100 cannot resolve that uncertainty.

**Smallest safe experiment.** Persist paired loss differences `d_i = loss_model,i - loss_market,i`. Estimate their dependence-aware standard error and calculate the sample needed for the registered minimum effect:

\[
n_{required}=\left(\frac{(z_{1-\alpha/2}+z_{1-\beta})s_d}{\delta_{min}}\right)^2.
\]

Here `s_d` must come from prior outer-fold paired differences, and `delta_min` must be fixed before 2026 outcomes are inspected.

**Immediate containment.** Treat Week 8 as descriptive only. Do not consume the one-time promotion test merely because both markets reach 100 rows.

**Durable fix.** Use a full-season or power-qualified gate requiring confidence bounds and practical effect sizes. Adjust the family-wise decision for two markets and multiple required metrics through a hierarchical gate: data integrity first, then probability skill, then calibration, then CLV and execution.

**Telemetry.** Effective sample size, paired-loss standard deviation, bootstrap design, confidence interval, minimum detectable effect, and number of hypotheses examined.

**Residual risk.** One NFL season may remain too small to prove an economically small edge. `INSUFFICIENT_EVIDENCE` must be a valid final state.

### Finding S4: publication-time leakage is not disproven

**Observed issue.** Every populated `source_updated_at_utc` value in games, team metrics, injuries, player usage, and model history is blank. Retrieval time is present, but it occurred in September 2026 for historical rows.

**Mechanism.** Event time, upstream publication time, revision time, and local retrieval time are different clocks. Lagging a statistic by one game prevents direct future-game leakage but does not prove the value or revision was available at the historical decision timestamp.

**Evidence.** Independent Python audit of authoritative exports; `DATA_DICTIONARY.md` explicitly permits blank upstream update times.[^1]

**Uncertainty.** Missing timestamps do not prove that leakage occurred. They prove that non-leakage cannot be audited from the exported evidence. Feature-specific risk depends on whether immutable source vintages can be recovered.

**Smallest safe experiment.** Select a stratified sample of historical games, reconstruct every feature from archived source snapshots if available, and compare it with the current backfilled value. Inject deliberately late records and verify they cannot enter a prediction before `available_at_utc`.

**Immediate containment.** Do not add injuries, weather, referee assignments, or late player news to historical backtests unless their publication vintages are recoverable.

**Durable fix.** Add `event_time_utc`, `published_at_utc`, `available_at_utc`, `retrieved_at_utc`, `revised_at_utc`, and `revision_id`. Model eligibility must use `available_at_utc <= decision_time_utc`.

**Telemetry.** Late-arrival rate, revision rate, source-clock coverage, age at decision, and features excluded for missing publication time.

**Residual risk.** Some public sports datasets are maintained as corrected current-state tables. If historical vintages do not exist, certain features can only be tested prospectively.

</statistical_validity>

<market_mechanics>

## Market mechanics

### Finding M1: decision price and closing price are the same record

**Observed issue.** Settlement selects the latest valid *prediction* between 90 and 5 minutes before kickoff and uses that prediction's Pinnacle line and probability as the closing benchmark. It then writes `line_clv = 0.0` and averages the field in reports.

**Mechanism.** A prediction snapshot can be a decision quote or a close, but it cannot be both for CLV measurement. This design can compare model probability with the contemporaneous market. It cannot determine whether a hypothetical bet beat a later, independently observed close.

**Evidence.** `src/nfl_bets/prospective.py`, lines 712–754, 764–895, and 1024–1047.[^5]

**Uncertainty.** The code-level conflation and synthetic zero are certain. The sign and magnitude of true CLV are wholly unknown because no independent closing observations exist.

**Smallest safe experiment.** For one pilot week, designate an immutable decision snapshot at each configured decision slot and capture a separate closing snapshot under a distinct `snapshot_purpose=CLOSE`. Reconcile both by game, book, market, side, and exact contract.

**Immediate containment.** Rename the current metric to `canonical_evaluation_snapshot`. Report CLV as `UNAVAILABLE`, never zero.

**Durable fix.** Store decision contract and independent close separately. Define contract CLV as closing-distribution expected return on the exact bet taken:

\[
CLV_{contract}=P_c(W)b_d-P_c(L),
\]

where `b_d` is net decimal payout at decision time and `P_c` is the no-vig closing distribution evaluated at the decision line. This incorporates price, line, and push probability. Also retain raw line movement and probability movement as diagnostics.

**Telemetry.** Capture success by purpose, quote age, line/price pair completeness, close-source fallback, key-number crossing, closing expected value, and missing-close reason.

**Residual risk.** A single sportsbook close can be stale, shaded, unavailable, or limit-constrained. A predefined multi-source hierarchy is still required.

### Finding M2: consensus construction creates an assumed and sometimes non-tradable object

**Observed issue.** Consensus line is `0.60 × Pinnacle + 0.40 × retail mean`. Probabilities are blended only when all books offer the same point. The 60/40 weight has no empirical estimate.

**Mechanism.** Averaging lines can create a number no sportsbook offered. Book errors are correlated because feeds and originators overlap. Equal retail averaging gives a stale or low-limit book the same retail influence as a sharper source. Liquidity and limits are absent.

**Evidence.** `src/nfl_bets/odds/consensus.py`, lines 187–220; handoff CSV row 21.[^2]

**Uncertainty.** The assumed 60/40 construction is certain. Its practical bias is unknown because the authoritative market tables are empty and no source-by-source closing or execution record exists.

**Smallest safe experiment.** During the observer pilot, score each book independently by quote completeness, staleness, subsequent movement, closing calibration, and available limit proxy. Compare Pinnacle-only, median line, best executable line, and registered weighted consensus.

**Immediate containment.** Keep the current consensus diagnostic-only. Never feed its synthetic line into stake or CLV calculations.

**Durable fix.** Separate three objects: `market_reference` for evaluation, `best_executable_contract` for decisions, and `market_state_features` for modelling. Learn weights only from past data and freeze them by market and season.

**Telemetry.** Book presence, stale-rate, dislocation from median, best-price availability, limit proxy, rejection rate, and source-specific calibration.

**Residual risk.** Public odds APIs do not necessarily represent the price or limit available to the actual account at execution.

### Finding M3: paid capture and evidence durability are not production-safe

**Observed issue.** The capture path can call the paid odds provider without first acquiring a durable, unique reservation for `(week_bucket, slot, request_kind)`. API-request rows use a random request ID rather than a scheduled-slot uniqueness key, and the database transaction begins in deferred mode. Raw odds, SQLite state, logs, and generated evaluations are ignored local artifacts with no tested backup/restore procedure. The scheduler appends to one log without rotation or failure alerting.

**Mechanism.** Two schedulers, a retry after an ambiguous timeout, or a crash between the provider call and local write can duplicate a paid call while leaving quota state incomplete. A disk or sync failure can then remove the only evidence needed to reproduce the prospective test. Unbounded logs can consume local storage and create a second failure path.

**Evidence.** `src/nfl_bets/pilot.py`, `src/nfl_bets/odds/client.py`, `src/nfl_bets/db.py`, scheduler scripts, `.gitignore`, and absence of a restore path.[^5] Local canonical guidance treats paid external calls as payment-like operations requiring stable idempotency and treats raw evidence as durable state that must be recoverable.[^6][^7]

**Uncertainty.** No duplicate charge or data-loss incident is recorded; the live tables are empty. This is a verified control gap and plausible failure mechanism, not a claim that failure has already occurred.

**Smallest safe experiment.** Against a stub provider, launch the same scheduled slot concurrently, inject timeouts before and after the simulated provider response, and require at most one outbound call and one reconcilable terminal record. Then back up raw odds, SQLite, logs, and evaluations; corrupt the originals; restore; and reproduce recorded hashes.

**Immediate containment.** Permit one manually supervised scheduler instance, disable automatic retries after ambiguous provider responses, snapshot local evidence before and after each pilot window, and alert on every non-successful capture.

**Durable fix.** Add a stable idempotency key with a database uniqueness constraint, acquire the reservation in an immediate transaction before the provider call, record attempt and reconciliation state, and define retry rules for each failure phase. Add versioned backups, retention, restore drills, bounded log rotation, disk-space alerts, and clean-clone CI.

**Telemetry.** Reservation conflicts, attempted and billable calls, ambiguous outcomes, quota reconciliation, last successful backup, restore age, hash verification, log size, disk space, scheduler lag, and alert delivery.

**Residual risk.** A provider can bill an ambiguous request even when the local client times out. The system therefore still needs provider-side request identifiers and manual reconciliation; local idempotency alone cannot guarantee exactly-once billing.

### Devig specification

For American odds `A`, implied probability is:

\[
q(A)=
\begin{cases}
\frac{|A|}{|A|+100}, & A<0\\
\frac{100}{A+100}, & A>0.
\end{cases}
\]

For a two-sided market with raw probabilities `q_1,q_2` and overround `Q=q_1+q_2>1`:

- Proportional normalization: `p_i=q_i/Q`.
- Additive: `p_i=q_i-(Q-1)/2`.
- Power: solve `q_1^k+q_2^k=1`, then `p_i=q_i^k`.
- Shin: test through a verified implementation, but recognize that for two outcomes Shin reduces algebraically to the additive method.[^15]

Shin's insider-trading rationale was developed for bookmaker pricing and favorite–longshot bias, not specifically for near-even NFL spread and total pairs.[^14] Harville does not devig a two-way spread or total; it derives ordered finish probabilities in multi-entry competitions.[^17]

**Production rule.** Keep proportional normalization as the transparent default until a past-only, book-specific comparison shows that another method improves out-of-sample log loss. Report sensitivity across proportional, additive/Shin, and power probabilities. Do not select a devig method on the prospective test.

### Liquidity and execution

A quoted edge is not executable edge without:

- sportsbook, account jurisdiction, line, price, maximum stake or limit proxy;
- timestamp and quote age at decision;
- acceptance, rejection, partial-fill, and price-change outcomes;
- expected slippage and delay;
- rules for pushes, voids, overtime, rescheduled games, and market grading;
- correlated exposure already held across the same game, team, week, and weather regime.

Until these fields exist, ROI backtests are frictionless simulations, not deployable results.

</market_mechanics>

<data_limitations>

## Data limitations and hidden variables

Feature expansion is not automatically improvement. Each source below needs a registered causal or predictive hypothesis, a publication-time contract, and incremental nested OOS value over a recalibrated market baseline.

| Priority | Missing or weak variable | Why it may matter | Leakage and modelling rule |
| --- | --- | --- | --- |
| P0 | Decision and independent closing market history | Required for CLV, executable edge, and market-regime analysis | Capture append-only decision and close snapshots. Never backfill a decision quote from a later close. |
| P0 | Publication and revision timestamps | Current exports cannot prove historical availability | Exclude any observation lacking a defensible `available_at_utc` from historical feature tests. |
| P1 | Starting quarterback and quarterback uncertainty | Quarterback changes can dominate spread and total repricing | Store source, announced time, probability of starting, expected snaps, and replacement distribution. Do not use final starter status retrospectively. |
| P1 | Offensive line, pass-rusher, and secondary availability | Cluster effects can change pressure, explosive plays, and substitution quality | Use expected snap loss and replacement quality with hierarchical shrinkage. Current injury history is too short and remains disabled. |
| P1 | Forecast-vintage weather | Wind, precipitation, temperature, and roof state can affect totals and kicking | Store forecast provider, issue time, valid time, location, uncertainty, and roof decision. Final observed weather is not a valid pregame backtest input.[^23] |
| P1 | Travel and schedule state | Time-zone change, international travel, short weeks, bye timing, and consecutive road games may alter preparation | Estimate era-specific effects and interactions. Do not hard-code “Thursday fatigue”; published NFL injury evidence is mixed.[^24] |
| P2 | Coaching and coordinator changes | Scheme priors can break across staff changes | Use explicit change points and preseason priors. Avoid subjective labels without reproducible coding. |
| P2 | Tactical structure | Personnel, motion, pressure, man/zone, and split-safety rates may explain matchup residuals | Coverage data is currently absent. Require lawful provenance and stable definitions before testing. |
| P3 | Referee crew | Penalty mix and enforcement may affect pace, drives, or margins | Assignment is late and sample per crew is small. Use partial pooling, multiple-testing control, and publication-time capture. Do not encode narratives or team-specific “bias.”[^25] |
| P3 | Market liquidity and limits | Separates informative prices from stale or recreational prices | Use observed or defensible proxies. Never equate sportsbook presence with liquidity. |

Historical NFL scores in this repository also reject a naive Poisson assumption. Across 4,431 completed regular-season games from 2009–2025, home score mean/variance are **23.67/104.69** and away score mean/variance are **21.65/98.15**. Variance is more than four times the mean for both. Home/away score correlation is **-0.0536**. Absolute final margins land on 3 in **14.58%** of games and on 7 in **8.76%**. These facts require an overdispersed, football-aware distribution with discrete key-number behavior and accord with prior NFL work that explicitly accommodates non-normal forecast errors.[^9]

</data_limitations>

<risk_and_sizing>

## Risk and sizing

### Single-bet economics with pushes

Let `P_W`, `P_P`, and `P_L` be model probabilities for win, push, and loss, summing to one. Let decimal odds be `d` and net profit per unit staked be `b=d-1`.

\[
EV=P_W b-P_L.
\]

Break-even conditional non-push probability is:

\[
p_{BE}=\frac{1}{d}, \qquad
p_{NP}=\frac{P_W}{P_W+P_L}.
\]

The push-aware full-Kelly fraction maximizes expected log wealth:[^18]

\[
g(f)=P_W\log(1+bf)+P_L\log(1-f)+P_P\log(1),
\]

which yields:

\[
f^*=\frac{bP_W-P_L}{b(P_W+P_L)}.
\]

If the numerator is non-positive, stake is zero.

### Parameter uncertainty

Raw Kelly assumes the probabilities are known. They are estimated and currently weakly validated. Research on Kelly under parameter uncertainty supports shrinking stakes rather than substituting a noisy point estimate as truth.[^19]

Define a registered uncertainty set `C_i` for each contract and execution cost `c_i`. Conservative edge is:

\[
EV_{i,lower}=\inf_{(P_W,P_P,P_L)\in C_i}(P_Wb_i-P_L)-c_i.
\]

Only if `EV_{i,lower}>0` may sizing continue. Then:

\[
f_i=\min\left(\rho f_i^*,\; cap_{bet},\; cap_{game}-exposure_{game},\; cap_{week}-exposure_{week}\right).
\]

`rho` is not automatically 0.25. Select it from prospective, uncertainty-aware bankroll simulations subject to a registered drawdown constraint. V1.1.2 must use `rho=0` because prospective calibration and uncertainty coverage do not exist.

### Correlated portfolio

For simultaneous wagers with scenario return vector `R_s` and stake vector `f`, solve:

\[
\max_f \sum_s w_s\log(1+f^TR_s)
\]

subject to non-negative wealth in every simulated scenario, per-bet caps, same-game caps, team/week exposure caps, and a drawdown-probability constraint. Risk-constrained Kelly provides a formal way to trade growth for drawdown control.[^20]

Spread and total bets on the same game are not independent. Neither are positions sharing quarterbacks, weather systems, divisional information, or the same market-originator error. A scalar Kelly calculation per bet will overstate total permissible exposure.

### Mandatory no-bet states

Stake must be exactly zero when any of the following holds:

- no independent close definition or executable decision price;
- missing or stale feature publication time;
- probability interval or calibration penalty makes conservative EV non-positive;
- quote age, book availability, price, or limit is unknown;
- model/version/hash mismatch;
- unmeasured correlation would breach an exposure cap;
- prospective calibration or uncertainty coverage has not passed;
- any required data-integrity or recovery check fails.

</risk_and_sizing>

<method_assessments>

## Method assessments

| Method | Disposition | Appropriate role | Principal reason |
| --- | --- | --- | --- |
| Bradley–Terry | **RETAIN AS BENCHMARK** | Dynamic latent team-strength baseline and optional feature | Correct for pairwise win propensity; insufficient for score magnitude, spreads, totals, and pushes.[^27] |
| Standard bivariate Poisson | **REJECT AS PRODUCTION DEFAULT** | Diagnostic benchmark only | NFL scores are strongly overdispersed, key-numbered, and slightly negatively correlated in this sample; the standard shared-component model has Poisson marginals and non-negative covariance.[^8] |
| Pythagorean expectation | **RETAIN AS LOW-DIMENSIONAL PRIOR** | Team-level scoring-strength/luck benchmark | Summarizes points for/against but discards opponent, market, timing, and game-level distribution. Fit its exponent in past-only data.[^10] |
| Elo | **REQUIRE AS BASELINE** | Simple dynamic team rating | Transparent, cheap, hard to overfit, and useful as a minimum challenger benchmark. |
| Glicko-2 | **TEST WITH CAUTION** | Uncertainty-aware rating benchmark | Rating deviation and volatility are useful, but official guidance expects moderate-to-large games per rating period; NFL teams play roughly one game per week.[^11] A state-space rating may fit the cadence better. |
| TrueSkill | **DEFER** | Player/lineup uncertainty research | Supports team composition and skill uncertainty, but reliable player participation, attribution, and timestamped lineup data are prerequisites.[^12] |
| Logarithmic loss | **REQUIRE** | Primary proper probability score with Brier | Strictly proper and sensitive to overconfidence. Must be paired, cross-fitted, and uncertainty-bounded.[^13] |
| Shin devig | **SENSITIVITY ONLY** | Comparison with proportional/additive/power methods | Its empirical advantages are mainly established in broader and multi-outcome markets; in a two-outcome market it collapses to additive.[^15][^16] |
| Harville | **REJECT** | None for spreads/totals | It models ordered outcomes in multi-entry competitions, not removal of vig from two-sided point markets.[^17] |
| Monte Carlo | **RETAIN AFTER DISTRIBUTION VALIDATION** | Correlated score, contract, and bankroll simulation | Simulation propagates assumptions; it cannot repair a misspecified distribution or create edge.[^21] |

### Bradley–Terry benchmark

For teams `i` and `j`, with latent strengths `r_i,r_j` and home effect `h`:

\[
P(i\text{ beats }j)=\operatorname{logit}^{-1}(r_i-r_j+h).
\]

Fit chronologically with partial pooling and season-to-season evolution. Score it against market-implied straight-up probability. It may feed a challenger as a pregame strength feature only if it adds nested OOS value. Do not convert its win probability linearly into a spread-cover probability.

### Elo benchmark

\[
E_i=\frac{1}{1+10^{-(R_i-R_j+H)/400}}, \qquad
R_i'=R_i+K(S_i-E_i).
\]

Home advantage `H`, update rate `K`, offseason regression, and any margin-of-victory multiplier must be selected inside past-only folds. The baseline should remain deliberately small.

### Pythagorean prior

\[
WinPct_i=\frac{PF_i^\gamma}{PF_i^\gamma+PA_i^\gamma}.
\]

Fit `gamma` only on earlier seasons. Shrink early-season points for and against toward league and roster priors. Use the result as a benchmark or prior, not as a direct spread/total price.

### Standard bivariate Poisson benchmark

Let independent latent counts be `U_1~Pois(λ_1)`, `U_2~Pois(λ_2)`, and `U_3~Pois(λ_3)`:

\[
X=U_1+U_3,\qquad Y=U_2+U_3.
\]

Then `Var(X)=E(X)`, `Var(Y)=E(Y)`, and `Cov(X,Y)=λ_3≥0`. Those restrictions conflict with this repository's score dispersion and slight negative correlation. A challenger should instead model the joint margin/total distribution using hierarchical location and scale models plus an empirical residual copula or another overdispersed distribution, followed by explicit discrete key-number correction.

### Monte Carlo use

Individual contract probabilities should use an analytic CDF or exact empirical distribution when available. Monte Carlo is justified for parameter uncertainty, correlated alternate lines, same-game portfolios, and bankroll paths. For an estimated probability `p` from `N` independent draws, simulation standard error is:

\[
SE_{MC}=\sqrt{\frac{p(1-p)}{N}}.
\]

Set `N` so simulation error is negligible relative to the registered betting threshold. Record seed, number of draws, parameter draw, residual source, and convergence diagnostics. Repeating a misspecified model one million times only estimates the wrong model more precisely.

</method_assessments>

<challenger_blueprint>

## Challenger blueprint

### Separation rule

V1.1.2 remains the frozen 2026 observer. The challenger receives a new model version, specification hash, feature manifest, decision policy, and prospective cutoff. If any 2026 outcome is used to design or select the challenger, 2026 cannot be its untouched test.

### Model stack

1. **Market-only baseline.** Decision-time line, both prices, no-vig probability under registered devig methods, book identity, time to kickoff, quote age, and market dispersion. Fit calibration only on prior periods.
2. **Small football baselines.** Elo, Bradley–Terry, and Pythagorean prior. Evaluate each alone and as an increment to market-only.
3. **Existing efficiency baseline.** Current EPA/success/pressure/continuity/rest ridge adjustment, evaluated through a fully nested probability pipeline.
4. **Distributional challenger.** Jointly model margin and total residual location, scale, tail behavior, and dependence. Preserve discrete score constraints and key-number mass through empirical or hurdle-style correction.
5. **Availability challenger.** Add quarterback and unit availability only after enough prospectively timestamped history exists.
6. **Weather/travel challenger.** Add forecast-vintage weather and schedule/travel effects through registered interactions, primarily for totals.
7. **Portfolio layer.** Enable simulation and robust Kelly only after predictive distribution and interval coverage pass.

### Preferred first challenger

Do not begin with a large nonlinear ensemble. Begin with a market-only calibrated baseline and a regularized distributional regression:

- margin residual location and total residual location as separate linear predictors;
- scale parameters allowed to vary by market total, favorite size, week, and selected pregame context;
- heavy-tailed or empirical residual marginals;
- dependence estimated from outer-training residuals;
- explicit mass adjustment at spread margins 3, 6, 7, 10, and 14;
- all transforms, strata, hyperparameters, mapper parameters, and calibrators fit inside the outer training period.

Only after this beats the market-only baseline should tree boosting, Bayesian nonlinear effects, or player-level TrueSkill be considered.

</challenger_blueprint>

<data_contracts>

## Future data contracts

### Market snapshot

Required fields:

`snapshot_id`, `snapshot_purpose`, `game_id`, `provider_event_id`, `market`, `side`, `sportsbook`, `line`, `american_price`, `decimal_price`, `observed_at_utc`, `book_updated_at_utc`, `retrieved_at_utc`, `quote_age_seconds`, `source`, `source_quality`, `limit_amount_or_proxy`, `is_executable`, `ruleset_id`, `content_hash`.

Invariants:

- `snapshot_purpose` is one of `DECISION`, `CLOSE`, or `DIAGNOSTIC`.
- A `DECISION` snapshot can never be relabelled as `CLOSE`.
- Both sides at the same point are required for same-point devig.
- Missing price, stale quote, invalid overround, or unmatched event fails closed.
- Scheduled paid capture has a stable idempotency key and a database uniqueness constraint.

### Feature observation

Required fields:

`feature_observation_id`, `game_id`, `entity_id`, `feature_name`, `value`, `unit`, `source`, `event_time_utc`, `published_at_utc`, `available_at_utc`, `retrieved_at_utc`, `revised_at_utc`, `revision_id`, `eligibility_cutoff_utc`, `content_hash`.

Invariant: a feature is eligible only when `available_at_utc <= decision_time_utc`. Retrieval time may not substitute for an unknown historical publication time.

### Prediction

Required fields:

`prediction_id`, `model_version`, artifact/spec/policy/data hashes, `decision_time_utc`, executable contract identity, `P_W`, `P_P`, `P_L`, conditional non-push probability, fair decimal/American price, interval or posterior quantiles, market probability under every registered devig method, point projection, distribution parameters, conservative EV, raw Kelly, adjusted Kelly, exposure state, and `NO_BET` reason.

Invariant: `P_W + P_P + P_L = 1` within numerical tolerance. No row may emit a positive stake when a required field is unavailable.

### Evaluation

Required fields:

decision contract, independent closing snapshot, result, model and market joint scores, conditional non-push scores, projection errors, `decision_line_clv`, `decision_price_clv`, key-number-aware line movement, closing contract EV, accepted/rejected execution outcome, slippage, stake, profit/loss, and correlation group identifiers.

</data_contracts>

<validation_protocol>

## Validation protocol

### Chronological nesting

For outer validation period `t`:

1. Use only observations available before `t`.
2. Generate inner chronological OOF predictions within the outer-training set.
3. Select features, hyperparameters, devig method, residual distribution, calibration method, and shrinkage from inner predictions only.
4. Refit the selected base model on the outer-training set.
5. Fit mapper and calibrator using training-side OOF predictions, never outer-validation outcomes.
6. Predict outer period `t` once.
7. Concatenate outer predictions for all reported development metrics.

Preprocessing and standardization are part of the fitted pipeline and must follow the same boundary.

### Required comparisons

- proportional no-vig market;
- past-only recalibrated market;
- Elo;
- Bradley–Terry;
- Pythagorean prior;
- current 21-feature ridge layer;
- each registered challenger;
- ablations by feature family.

### Scoring

For binary conditional non-push outcome `y_i` and prediction `p_i`:

\[
LogLoss=-\frac{1}{n}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)],
\]

\[
Brier=\frac{1}{n}\sum_i(p_i-y_i)^2.
\]

For spread and integer-total contracts, also score the full three-outcome vector with multinomial log loss and multiclass Brier score so push-mass quality cannot hide behind push exclusion.

Report:

- paired model-minus-market score differences;
- relative skill scores;
- hierarchical season/week block-bootstrap intervals;
- calibration-in-the-large, calibration slope, flexible calibration curve, and uncertainty intervals;
- reliability and resolution, with bins used only for visualization;
- line/price CLV from an independent close;
- ROI, turnover, drawdown, and rejection/slippage sensitivity only after a fixed bet rule exists.

### Adversarial tests

- Deliberately move a source publication timestamp after decision time and require exclusion.
- Inject a future result into a lag store and require leakage validation to fail.
- Prove no scored row was used by its standardizer, selector, mapper, or calibrator.
- Duplicate and concurrently launch a paid scheduled slot; require one durable reservation and at most one provider call.
- Supply stale, one-sided, mismatched-point, suspended, zero-overround, and conflicting quotes; require no prediction or no-bet state as specified.
- Test pushes at integer totals and margins 3, 6, 7, 10, and 14.
- Change devig method and verify the sensitivity report changes while the registered production method remains fixed.
- Remove decision or closing price and require CLV to be unavailable, never zero.
- Reconcile price CLV, line CLV, and closing-contract EV independently; crossing a key number must not be treated as an ordinary fractional-point move.
- Feed missing, stale, revised, duplicated, and conflicting feature and quote records through eligibility checks; require the specified fail-closed or no-bet state.
- Run predeclared regime diagnostics for quarterback status, injury clusters, forecast weather, rest, travel, coaching changes, playing surface and roof state, referee crew, and liquidity tier. Use them to detect fragility, not to rescue a failed aggregate test.
- Run rolling-origin era-drift checks for coefficient stability, residual scale, calibration, feature-definition versions, and market-relative loss.
- Correlate spread and total outcomes in bankroll scenarios and verify exposure caps, drawdown constraints, and ruin-probability limits bind.
- Corrupt raw odds, SQLite, and generated reports; restore from backup and reproduce hashes.
- Run from a clean clone in CI and compare candidate, specification, feature, and report hashes.

</validation_protocol>

<promotion_gates>

## Promotion gates

Promotion is hierarchical. Failure or insufficiency at any stage stops the process.

### Gate 0: integrity and reproducibility

- clean-clone build, tests, lint, types, validation, and deterministic artifact hashes pass;
- decision and close snapshots are distinct and recoverable;
- publication-time coverage meets the registered feature policy;
- duplicate paid capture, crash recovery, backup restore, and log/alert tests pass.

### Gate 1: sample sufficiency

- minimum sample size is calculated from registered `delta_min`, paired-loss variance, desired power, and dependence adjustment; validation sample size must target precision rather than rely on a generic event count;[^22]
- Week 8 and 100 non-push observations are descriptive checkpoints only;
- if the power-qualified sample is unavailable, status is `INSUFFICIENT_EVIDENCE`, not pass or fail.

### Gate 2: predictive skill

For both spreads and totals:

- model Brier and log loss beat the past-only recalibrated market baseline;
- the 95% interval for paired model-minus-market loss lies below zero;
- point estimate reaches a registered minimum practical improvement;
- full three-outcome scoring does not reveal degraded push modelling;
- no unregistered subgroup or feature search is used to rescue a failed aggregate result.

Default minimum practical Brier skill is **0.5% relative improvement over market** unless pre-2026 power analysis justifies a different value. This corresponds to roughly 0.00125 when market Brier is near 0.25, substantially larger than the current optimistic development improvements.

### Gate 3: calibration

- calibration intercept point estimate is within ±0.05;
- calibration slope point estimate is within 0.80–1.20;
- intervals include ideal values 0 and 1 without being so wide that the criterion is uninformative;
- registered interval or posterior predictive coverage passes by market and key-number stratum where sample size permits.

The existing ±0.20 intercept and 0.50–1.50 slope bands are too permissive for staking.

### Gate 4: market and execution

- execution-adjusted closing contract EV is positive with its 95% lower bound above zero;
- results survive registered devig sensitivity, stale-quote exclusions, book hierarchy, and plausible slippage;
- executable price and limit coverage are sufficient for the intended stake scale;
- no single book, week, team, or key-number subset supplies the entire result.

### Gate 5: bankroll safety

- conservative EV remains positive after probability and execution uncertainty;
- risk-constrained or fractional Kelly policy passes registered maximum drawdown and ruin-probability thresholds;
- correlated same-game and shared-factor exposure caps pass stress tests;
- a human-controlled kill switch and zero-stake fallback are tested.

Passing Gates 0–4 authorizes only a separately reviewed staking design. It does not automatically authorize wagering.

</promotion_gates>

<implementation_sequence>

## Ordered implementation sequence

| Order | Change | Expected benefit | Cost | Leakage risk | V1.1.2 freeze impact |
| ---: | --- | --- | --- | --- | --- |
| 1 | Tag the complete observer application separately from the V1.1.2 model freeze and reconcile the divergent branch | Clear model versus application provenance | Low | None | Does not alter candidate; creates an application release boundary |
| 2 | Enforce stable scheduled-slot idempotency in the capture transaction | Prevent duplicate provider charges and quota races | Low–medium | None | Observer infrastructure only |
| 3 | Capture independent decision and closing snapshots; replace hard-coded CLV | Makes market mechanics and CLV measurable | Medium | Low if specified before observations | Observer schema change; candidate math unchanged |
| 4 | Add backup/restore, log rotation, alerts, and clean-clone CI | Makes evidence durable and failures visible | Medium | None | Observer infrastructure only |
| 5 | Implement nested mapper/calibrator evaluation and paired uncertainty | Removes optimistic development scoring | Medium | High if tuned on 2026; use pre-2026 only | New analysis report, not V1.1.2 mutation |
| 6 | Add market-only, Elo, Bradley–Terry, and Pythagorean benchmarks | Establishes whether complexity adds information | Medium | Moderate; register variants | New challenger specification |
| 7 | Build the small distributional margin/total challenger | Tests an NFL-appropriate joint distribution | Medium–high | High; freeze formula and strata before test | New model version required |
| 8 | Begin prospective quarterback, injury, forecast-weather, travel, and referee data collection | Creates leakage-safe future feature history | Medium–high | Low prospectively, high if backfilled | Data collection only until a later challenger |
| 9 | Add Monte Carlo and robust portfolio Kelly after calibration passes | Converts validated distributions into bounded exposure | High | High if optimized on realized ROI | Separate staking-policy freeze required |

### Next smallest reversible action

Fix the market experiment before changing the predictive model: create distinct `DECISION` and `CLOSE` snapshot purposes, make CLV unavailable until both exist, and enforce one paid capture per scheduled slot. The validating signal is a one-week pilot in which every eligible game has independently timestamped decision and close contracts, no duplicate provider call, no synthetic zero CLV, and full restoration from backup.

</implementation_sequence>

<residual_risks>

## Residual risks

- The NFL regular season provides a small annual sample relative to plausible edges. A single season may not resolve effects of the size currently reported.
- Sportsbook prices, limits, and account access can change. Public API quotes may not equal executable account contracts.
- Market-originator and retail-book errors are correlated. More books do not create independent evidence.
- Feature definitions such as EPA, pressure proxies, and tracking-derived coverage can change across eras or upstream revisions.
- Player availability and weather are high-value candidates but have severe publication-time and missingness risks.
- A model can beat Brier or log loss without producing bets after vig and friction. Conversely, realized short-run ROI can be positive without predictive skill.
- Kelly sizing is fragile to probability error and correlation misspecification. Even a promoted model requires conservative caps and a kill switch.
- No formula listed here is presumed superior. Each is a registered challenger or benchmark until it earns incremental chronological OOS value.

The intentionally accepted residual risk is scientific failure: the completed prospective program may conclude that the market cannot be beaten with the available data. That outcome is preferable to deploying a false edge.

</residual_risks>

<sources>

## Sources

[^1]: NFL-2026-BETS local project evidence. `PROJECT_STATE.md`, `DATA_DICTIONARY.md`, `MODEL_SPEC_V1_1.md`, `STAGE_9_PROSPECTIVE_LEDGER.md`, authoritative root CSVs, and SQLite state. Accessed 2026-09-14.
[^2]: `NFL_2026_Model_Technical_Review_Handoff.csv`, 38 evidence and review-agenda rows. Accessed 2026-09-14.
[^3]: NFL-2026-BETS, `reports/model_1.1.2_development.json` and `reports/model_1.1.2_development_folds.csv`. Independent calculations performed with Python on 2026-09-14.
[^4]: NFL-2026-BETS, `reports/model_1.0.1_test_2025.json`. Untouched 2025 V1.0.1 evaluation.
[^5]: NFL-2026-BETS source inspection: `src/nfl_bets/model/v11.py`, `src/nfl_bets/prospective.py`, `src/nfl_bets/pilot.py`, `src/nfl_bets/odds/client.py`, `src/nfl_bets/odds/consensus.py`, and `src/nfl_bets/db.py`. Commit `1313bd7`.
[^6]: Local canonical corpus, `canonical/01-high-level-patterns.md`, durable/derived state and payment-like idempotency pattern.
[^7]: Local canonical corpus, `canonical/02-deep-theory-and-math.md`, transactions, provenance, recovery-friendly derivation, and retry amplification; `canonical/04-real-world-failures.md`, backup/data-loss and unbounded-log mechanisms; `canonical/06-secure-reliable-systems.md` and `canonical/07-application-security-controls.md`, recovery, threat modelling, secrets, APIs, logging, and supply-chain controls.
[^8]: Dimitris Karlis and Ioannis Ntzoufras. “[Analysis of Sports Data by Using Bivariate Poisson Models](https://doi.org/10.1111/1467-9884.00366).” *The Statistician* 52(3), 2003, 381–393.
[^9]: Michael Cain, David Law, and David A. Peel. “[Testing for Statistical and Market Efficiency When Forecast Errors Are Non-normal: The NFL Betting Market](https://doi.org/10.1002/1099-131X(200012)19:7%3C575::AID-FOR765%3E3.0.CO;2-U).” *Journal of Forecasting* 19(7), 2000, 575–586.
[^10]: Cary A. Caro and Ryan Machtmes. “[Testing the Utility of the Pythagorean Expectation Formula on Division One College Football](https://doi.org/10.19030/jber.v11i12.8261).” *Journal of Business & Economics Research* 11(12), 2013, 537–542. Used as evidence for the method's team-season role, not as NFL betting validation.
[^11]: Mark E. Glickman. “[Example of the Glicko-2 System](https://www.glicko.net/glicko/glicko2.pdf).” Boston University, revised March 22, 2022.
[^12]: Ralf Herbrich, Tom Minka, and Thore Graepel. “[TrueSkill: A Bayesian Skill Rating System](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf).” *Advances in Neural Information Processing Systems 19*, 2007.
[^13]: Tilmann Gneiting and Adrian E. Raftery. “[Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).” *Journal of the American Statistical Association* 102(477), 2007, 359–378.
[^14]: Hyun Song Shin. “[Optimal Betting Odds Against Insider Traders](https://doi.org/10.2307/2234434).” *The Economic Journal* 101(408), 1991, 1179–1185.
[^15]: Stephen Clarke, Stephanie Kovalchik, and Martin Ingram. “[Adjusting Bookmaker's Odds to Allow for Overround](https://doi.org/10.11648/j.ajss.20170506.12).” *American Journal of Sports Science* 5(6), 2017, 45–49.
[^16]: Erik Štrumbelj. “[On Determining Probability Forecasts from Betting Odds](https://doi.org/10.1016/j.ijforecast.2014.02.008).” *International Journal of Forecasting* 30(4), 2014, 934–943.
[^17]: David A. Harville. “[Assigning Probabilities to the Outcomes of Multi-Entry Competitions](https://doi.org/10.1080/01621459.1973.10482425).” *Journal of the American Statistical Association* 68(342), 1973, 312–316.
[^18]: J. L. Kelly Jr. “[A New Interpretation of Information Rate](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x).” *Bell System Technical Journal* 35(4), 1956, 917–926.
[^19]: Rose D. Baker and Ian G. McHale. “[Optimal Betting Under Parameter Uncertainty: Improving the Kelly Criterion](https://doi.org/10.1287/deca.2013.0271).” *Decision Analysis* 10(3), 2013, 189–199.
[^20]: Enzo Busseti, Ernest K. Ryu, and Stephen Boyd. “[Risk-Constrained Kelly Gambling](https://web.stanford.edu/~boyd/papers/kelly.html).” 2016.
[^21]: Nicholas Metropolis and S. Ulam. “[The Monte Carlo Method](https://doi.org/10.1080/01621459.1949.10483310).” *Journal of the American Statistical Association* 44(247), 1949, 335–341.
[^22]: Richard D. Riley et al. “[Minimum Sample Size for External Validation of a Clinical Prediction Model with a Binary Outcome](https://doi.org/10.1002/sim.9025).” *Statistics in Medicine* 40(19), 2021, 4230–4251. Used for the general principle that validation sample size must target precision of calibration and performance estimates, not as an NFL-specific rule.
[^23]: Rodney J. Paul. “[The Impact of Atmospheric Conditions on Actual and Expected Scoring in the NFL](https://doi.org/10.1177/155862351701200102).” *International Journal of Sports Finance* 12(1), 2017.
[^24]: T. Sean Castle et al. “[The Effect of Thursday Night Games on In-Game Injury Rates in the National Football League](https://doi.org/10.1177/0363546520919989).” *American Journal of Sports Medicine* 48(6), 2020.
[^25]: Michael J. Lopez. “[Persuaded Under Pressure: Evidence from the National Football League](https://doi.org/10.1111/ecin.12341).” *Economic Inquiry* 54(4), 2016, 1763–1773.
[^26]: Philip K. Gray and Stephen F. Gray. “[Testing Market Efficiency: Evidence from the NFL Sports Betting Market](https://doi.org/10.1111/j.1540-6261.1997.tb01129.x).” *Journal of Finance* 52(4), 1997, 1725–1737.
[^27]: Ralph Allan Bradley and Milton E. Terry. “[Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://doi.org/10.1093/biomet/39.3-4.324).” *Biometrika* 39(3/4), 1952, 324–345.

</sources>

</quant_model_review>
