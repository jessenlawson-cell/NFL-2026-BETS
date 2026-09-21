# Sequence 2 — Scheduled Capture Idempotency and Paid-Call Reconciliation

## Delivered boundary

Configured scheduled captures now use one stable key:

`scheduled-capture:v1:<Tuesday week bucket>:<slot>:full-board:<purpose>`

SQLite enforces uniqueness on the underlying (`week_bucket`, `slot`, `request_kind`) tuple.
The reservation is committed with `BEGIN IMMEDIATE` before client construction or provider
contact. A second process records a reservation conflict and exits without calling the provider.
The purpose is the slot's registered lowercase `decision` or `close` role. Manual captures remain
outside the scheduled-pilot keyspace.

Migration backfills any legacy scheduled request into a reservation before new captures are
allowed. Existing raw-snapshot or HTTP-response evidence conservatively establishes the legacy
call/response markers; ambiguous legacy attempts remain blocked for provider review.

Each provider attempt is append-only in `api_requests`. The application commits these boundaries
separately: reservation, provider call started, response received, raw response saved, parsed, and
complete. Sanitized `x-request-id` or `request-id` is retained when supplied. Secret values and
unsafe response headers are never persisted. Every operator resolution is appended to
`capture_reconciliations`; the latest reservation state does not erase earlier decisions or notes.

## Retry and reconciliation rules

| Last durable evidence | Required state | Provider retry |
| --- | --- | --- |
| Reservation exists; provider call never started | Operator records `SAFE_TO_RETRY_NO_CALL` with evidence | One explicitly authorized attempt |
| Call started; no response was durably recorded | `PROVIDER_CHECK_REQUIRED` | Blocked until provider confirms billing outcome |
| Provider confirms ambiguous call was not billed | Operator records `PROVIDER_CONFIRMED_NOT_BILLED` | One explicitly authorized attempt |
| Provider confirms the call was billed | Operator records `PROVIDER_CONFIRMED_BILLED` | Never |
| Response received or raw body saved | `NO_RETRY_RESPONSE_RECEIVED` or `NO_RETRY_RAW_SAVED` | Never; repair from saved/local evidence |
| HTTP, parsing, or prediction failure after raw save | Closed without provider retry | Re-run the downstream local step only |
| Complete | `COMPLETE` | Never |

There is no time-based lease takeover and no automatic retry. A stale `RESERVED` or `IN_FLIGHT`
state fails closed until an operator verifies the worker state and, when a call may have started,
checks the provider ledger.

## Operations

Inspect a slot without provider contact:

```powershell
docker compose run --rm nfl-bets pilot reconcile --slot monday_1200 `
  --week-bucket 2026-09-15
```

Record a resolution only after verifying the evidence:

```powershell
docker compose run --rm nfl-bets pilot reconcile --slot monday_1200 `
  --week-bucket 2026-09-15 `
  --resolution PROVIDER_CONFIRMED_NOT_BILLED `
  --note "Provider ledger checked; request was not billed."
```

Do not place API keys, credentials, or account data in reconciliation notes.

## Verification contract

Automated tests launch the same slot concurrently and require one reservation, one outbound call,
one terminal record, and a durable conflict count. Additional tests inject client setup failure,
an ambiguous timeout, and raw-persistence failure. They verify that retries remain blocked unless
the recorded evidence makes one explicitly safe. Production validation reconciles reservation
keys, active attempts, provider-call counts, and retry authorization states.

V1.1.2 remains `LOCKED_UNTESTED_2026`, observer-only, and PASS-only. This change does not alter its
artifact, specification, features, calibration, probability mapping, prospective cutoff, or test
population. Sequence item 3 is documented separately in `SEQUENCE_3_MARKET_SNAPSHOTS_CLV.md`.

## Acceptance record

Verified at `2026-09-14T18:40:47.1240268Z` from the saved local project, based on
`observer-app-v1.0.0`:

- full automated suite: 52 tests passed;
- Ruff: passed;
- strict mypy: passed for 23 source files;
- production validation: `VALID`, including reservation/attempt/call reconciliation;
- frozen V1.1.2 artifact, metadata, normalized specification, feature input, development-feature
  identity, and prospective cutoff: unchanged;
- provider calls during implementation and testing: zero real calls; all adversarial calls used
  an in-process stub.

The live migration found one pre-existing `monday_1200` request in week bucket `2026-09-08` with a
saved raw HTTP 401 response. It was backfilled as `CLOSED_NO_RETRY`; its evidence was preserved and
this task did not contact the provider or retry it.
