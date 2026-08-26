---
id: TASK-22215
title: Stagger the boot-time background worker fleet
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 10:10'
labels:
  - performance
  - startup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22215).

Boot-time concurrent workers went 4 -> 7 since the pin (new: chachanotes-fts-backfill,
the initial-screen pre-import thread, actor-pack recovery relocation). Under the GIL these
CPU-bound import/tokenize threads plus the Textual pump share one interpreter during the
first seconds after mount — worst on the first post-upgrade boot when 22200's backfill
runs to completion alongside the pre-importer. Each worker is individually justified; the
aggregate is what the user feels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Boot workers are census'd (a test pins the set, so an eighth is a reviewed decision) and started with an explicit priority/stagger policy
- [x] #2 Input latency during the first 5 s after mount measured before/after on a warm boot and on a simulated first-post-upgrade boot
- [x] #3 Backfills yield to foreground work (coordinate with TASK-22200)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enumerate the boot fleet from the 22222 census: for each worker, what user-visible surface waits on it (starvation check before any reorder).
2. Add a pure policy module (Utils/boot_worker_policy.py): ordered specs (key/name/group/tier/unblocks) + a concurrency-capped admission gate. Red-first policy test.
3. Wire app.py: the two FTS backfills leave on_mount (pre-first-paint) for the staggered tier; actor-pack recovery + staging sweep join it; admission released from on_worker_state_changed with a reconcile watchdog so a lost event cannot strand a pending worker; nothing admitted once _shutting_down.
4. AC3: re-verify 22200's ChaChaNotes pacing (already done, do not duplicate); close the remaining gap on the subscriptions FTS backfill (no pacing, no abort poll today).
5. Measure AC2: headless Pilot boot, loop-side heartbeat + synthetic keypress latency for the first 5 s, warm boot AND simulated first-post-upgrade boot (seeded history with messages_fts cleared). A/A control first, arms interleaved in both orders.
6. Targeted tests + --collect-only sweep, preflight, mutation test (break the order/cap -> policy test reds), teardown walk (quit inside the staggered window).
7. Census stays green or is updated deliberately with the reason recorded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped: an explicit boot-worker start policy (`Utils/boot_worker_policy.py`) with a
serial admission gate, the two FTS backfills moved off `on_mount` (they used to start
before first paint), and the subscriptions backfill given the pacing its ChaChaNotes
sibling already had. Measured: a real win on the post-upgrade boot shape, an honest wash
on the warm one.

### What each boot worker unblocks (the starvation check, done before any reorder)

| worker (name, group) | tier | what waits on it |
|---|---|---|
| `run`, `scheduling` | immediate | overdue reminders / scheduled watchlist checks. Coroutine worker that spends its life awaiting; must stay on the app's one loop (the watchlists in-flight guard is lock-free on that). |
| `restore_ingest_jobs`, `ingest_restore` | immediate | the Library ingest job history. Until it lands the registry is EMPTY — a wrong answer, not a slow one. |
| `deferred_actor_pack_recovery` | staggered #1 | nothing hard: Personas' first library read and `create_persona` gate on the coordinator's own once-lock. The worker is the prefetch that keeps that gate off the event loop. |
| `deferred_actor_pack_staging_sweep` | staggered #2 | nothing hard: `inspect_archive` gates on the same once-lock. Prefetch of a filesystem walk. |
| `_backfill_chachanotes_messages_fts` | staggered #3 | nothing. Search fills in progressively; frontier is `messages_fts_docsize` in the DB. Longest member on a post-upgrade boot. |
| `_backfill_subscription_items_fts` | staggered #4 | nothing. Same shape against `subscription_items_fts_docsize`. |

Deliberately NOT gated: the screen pre-importer (a daemon thread, already proportionally
paced by TASK-22214, and the thing that protects the first click to Library/Settings —
queueing it behind a minutes-long backfill would trade away its whole purpose), and
`on_mount`'s await-shaped research/scheduler coroutine workers.

### The policy

Order = the table above. Cap = **1** (`MAX_CONCURRENT_STAGGERED_BOOT_WORKERS`), so the
order IS the schedule. The first cut used 2 and was wrong for a specific reason worth
recording: with two slots the short prefetches finish immediately and then BOTH
whole-table re-tokenizations are admitted — the exact worst shape the finding is about.
Admission advances on `Worker.StateChanged` (the app-wide hook already sees every
transition); a slow `set_interval` reconcile is the backstop for a terminal transition
that never arrives, and it stops itself once the gate drains. `_shutting_down` closes the
gate, so a completion landing mid-teardown cannot start a fresh thread worker.

### AC #2 — measured (headless Pilot boot; loop-side 20 ms heartbeat + `pilot.press` every 100 ms for 5 s from `_ui_ready`; fresh copy of a seeded 30k-message / 20k-item profile per run; arms interleaved in BOTH orders; A/A control at the same n)

Post-upgrade shape (both FTS indexes emptied, i.e. the v45->v46 window), n=8/arm:

| metric | before | after | A/A control (n=4) |
|---|---|---|---|
| worst keypress | **625-876 ms** | **395-467 ms** | 426-476 ms |
| mean keypress | **111-124 ms** | **99-109 ms** | 101-104 ms |
| median keypress | 73-93 ms | 72-92 ms | 72-90 ms (bimodal) |
| p95 keypress | 125-169 ms | 125-183 ms | 136-177 ms |
| loop excess stall, 5 s | 566-613 ms | 570-625 ms | 566-658 ms |
| time to `_ui_ready` | 1.39-1.52 s | 1.36-1.49 s | — |

Worst-keypress and mean ranges are disjoint between arms AND the control sits inside the
after range, so those two carry the claim. **The median does not**: the A/A control
produced both modes with the code held constant, which is what stopped a "-22% median"
headline from being written (lesson filed in `lessons-testing-evidence.md`). p95, loop
stall and time-to-ready are washes.

Warm boot (same profile, indexes complete): **wash on every metric** (median 71-72 both
arms, worst 389-434 before vs 425-452 after against an A/A of 416-458, stall 544-616 vs
559-616). Expected — with nothing to backfill the fleet is four cheap calls in both arms,
and on an M-series the loop is not starved either way (the TASK-21113/22214 precedent).

Anti-vacuity, read off the profile at the end of each window: both arms indexed ~21k
messages during the 5 s, so a live backfill really was competing in both. The mechanism
of the win is visible in the same numbers: `subscription_items` indexed during the window
= **20000 before** (the second re-tokenization ran concurrently) vs **0 after** (queued
behind the first). A 25 s run confirms deferral is not abandonment: 30000/20000 rows
indexed, gate fully drained.

### AC #3 — 22200 re-verified, its gap closed

The ChaChaNotes backfill's pacing was already correct and is untouched (its 9 pacing
tests stay green): inter-chunk pause, lock-queue backoff, abort-sliced sleeps, worker
`is_cancelled` wired through. The **subscriptions** backfill had none of it — a
back-to-back chunk loop on `subscriptions.db`, where every watchlist ingest/status/
briefing write is also `BEGIN IMMEDIATE`, and no cancellation poll at all. It now shares
the primitives (extracted verbatim into `DB/fts_backfill_pacing.py`) and `app.py` hands it
the worker's cancellation flag. Its lock-retry schedule was deliberately NOT copied —
sizing a retry loop without a measurement is the clever-and-unstable half of the owner's
ruling.

### Teardown walk (quit inside the staggered window)

Quit with the ChaChaNotes backfill in flight and the subscriptions one pending:
`on_shutdown_request` returned in 1 ms, the gate closed, the pending worker never started,
the interrupted run left 4000 rows indexed (consistent, resumable), and the next boot
finished both (30000/20000) and drained the gate.

### Census

`Tests/Performance/test_boot_worker_census.py` stays green with **no allowlist change**:
all four staggered members keep their (name, group) identity and merely move to the
post-`_ui_ready` tier, still inside the census's settle window. Its docstring now names
the policy module, and the new policy test cross-checks every policy row against the
allowlist so the two files cannot drift.

### Mutation tests (all restored)

1. cap 1 -> 4: `test_the_concurrency_cap_is_below_the_staggered_fleet_size` red.
2. app ignores the policy cap: `test_deferred_startup_starts_the_cap_then_advances_on_completion` + `test_shutdown_closes_the_gate_instead_of_starting_more_work` red.
3. staggered order reversed: `test_staggered_order_runs_prefetches_before_the_resumable_backfills` red.
4. subscriptions pacing off: `test_backfill_sleeps_the_configured_pause_between_chunks` + `test_abort_cuts_an_in_flight_pause_at_the_poll_slice` red.
Plus the born-red evidence for the tier move: before the change,
`test_every_staggered_body_runs_after_the_ui_is_ready` failed with *"staggered boot
workers that ran before _ui_ready: ['chachanotes_fts_backfill',
'subscriptions_fts_backfill']"*.

### Files

Added `tldw_chatbook/Utils/boot_worker_policy.py`, `tldw_chatbook/DB/fts_backfill_pacing.py`,
`Tests/App/test_boot_worker_stagger_policy.py`. Modified `tldw_chatbook/app.py`,
`tldw_chatbook/Subscriptions/fts_backfill.py`, `tldw_chatbook/DB/chachanotes_fts_backfill.py`
(pacing primitives re-exported from the shared module; semantics unchanged),
`Tests/Subscriptions/test_fts_backfill.py`, `Tests/UI/test_console_runtime_ownership.py`
(its `getsource` pin followed the recovery kick to the new seam),
`Tests/Performance/test_boot_worker_census.py` (docstring), `backlog/docs/lessons-testing-evidence.md`,
`Docs/security/production-diagnostic-inventory.json` (7 added log lines, each reviewed:
policy keys and row counts only — no user content, secrets, paths or URLs).
<!-- SECTION:NOTES:END -->
