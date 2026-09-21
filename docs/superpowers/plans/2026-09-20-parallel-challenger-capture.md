# Parallel Challenger and Automated Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development task-by-task.

**Goal:** Freeze a no-stakes `challenger-0.2.0`, score the same snapshots as frozen `1.1.2`, automate the full September 22 pilot, and provide a blinded challenger chat.

**Architecture:** Extend the existing challenger and capture pipeline. Keep one immutable market ledger, preserve the frozen benchmark, and append only PASS/shadow predictions. Use current dependencies and existing feature/probability machinery.

**Tech Stack:** Python 3.13, Polars, pandas, NumPy, scikit-learn, SQLite, Typer, PowerShell, Docker Compose, Windows Task Scheduler.

**Spec:** `CHALLENGER_SPEC.md`

## Global Constraints

- Never modify or retrain model `1.1.2`.
- Never retry an ambiguous odds-provider call automatically.
- Never expose outcomes, CLV, ROI, settlement, or comparison before Week 8.
- Never generate stakes; every official decision remains PASS.
- Preserve all unrelated dirty and untracked user files.
- Do not freeze the challenger until every 2026 Week 2 game is final.

## Review Focus

- Nested selection must never use its validation season during model or calibration fitting.
- A challenger scoring failure after capture must not cause a second provider call.
- CLOSE snapshots must never create either lane's predictions.
- Latest-snapshot resolution must reject non-DECISION and post-kickoff contracts.
- Windows pilot tasks must use exact concrete dates and have no restart/retry policy.

---

### Task 1: Deterministic challenger selection and evidence

**Files:**
- Modify: `src/nfl_bets/model/challenger.py`
- Modify: `src/nfl_bets/challenger.py`
- Modify: `CHALLENGER_SPEC.md`
- Test: `tests/unit/test_challenger.py`

**Interfaces:**
- Preserve `model_predictions`; map football delta to `raw_adjustment`, football weight to `adjustment_weight`, and blended estimate to `final_projection`.
- Produce per-market selected features/model family, nested metrics, bootstrap intervals, calibration, edge monotonicity, disagreement, and drift metadata.

- [ ] Add failing tests for deterministic Ridge/tree/blend candidates, per-market selection, nested boundaries, bounded market weights, and evidence fields.
- [ ] Run focused tests and confirm the expected failures.
- [ ] Implement the smallest compatible challenger artifact and report changes.
- [ ] Run focused tests and the full suite.

### Task 2: Latest snapshot and dual-lane orchestration

**Files:**
- Modify: `src/nfl_bets/cli.py`
- Modify: `src/nfl_bets/pilot.py`
- Test: `tests/unit/test_challenger.py`
- Test: `tests/unit/test_pilot.py`

**Interfaces:**
- Add `challenger predict --latest --top N` without provider contact.
- Add optional `pilot capture --shadow-version challenger-0.2.0`; persist a single capture and isolate shadow-scoring failure from provider retry behavior.

- [ ] Add failing tests for latest DECISION resolution, CLOSE rejection, shared snapshot IDs, and isolated shadow failure.
- [ ] Run focused tests and confirm the expected failures.
- [ ] Implement the CLI and orchestration changes.
- [ ] Run focused tests and the full suite.

### Task 3: One-time and recurring Windows automation

**Files:**
- Create: `scripts/install_pilot_tasks.ps1`
- Create: `scripts/finalize_pilot_tasks.ps1`
- Modify: `scripts/run_scheduled_slot.ps1`
- Modify: `scripts/install_scheduled_tasks.ps1`

**Interfaces:**
- Install the 16 concrete September 23-28 pilot tasks plus a September 28 20:15 finalizer.
- Pass both model versions to the runner; use `IgnoreNew`, no automatic restart, and current-user execution.

- [ ] Add a dry-run mode that emits the exact task manifest without registration.
- [ ] Verify the dry-run manifest against the approved dates, slots, purposes, and commands.
- [ ] Register the one-time tasks only after model/code verification.
- [ ] Run zero-credit preflight and inspect registered task state.

### Task 4: Freeze gate, blind chat, and final verification

**Files:**
- Modify: `PROJECT_STATE.md` only after the verified freeze.
- Append immutable artifact/manifests only after every Week 2 game is final.

**Interfaces:**
- Freeze `challenger-0.2.0` through Week 2 and keep it immutable through Week 8.
- Create and pin `NFL Challenger — Weeks 3–8 Shadow Predictions` in the existing NFL project, using snapshots only and hiding realized results.

- [ ] Verify Week 2 finality, refresh shared observations, validate, and freeze or fail closed.
- [ ] Compare protected `1.1.2` hashes before and after.
- [ ] Run pytest, Ruff, mypy, validation, serialization reload, and `git diff --check`.
- [ ] Create the guarded local challenger chat and verify its title/project placement.
