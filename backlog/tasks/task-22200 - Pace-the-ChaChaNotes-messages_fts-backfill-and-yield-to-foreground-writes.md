---
id: TASK-22200
title: >-
  Pace the ChaChaNotes messages_fts backfill and yield to foreground writes
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
labels:
  - performance
  - database
  - console
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22200).

`DB/chachanotes_fts_backfill.py:82-97` drives `backfill_messages_fts` in a tight loop with
no sleep, yield, or time budget between chunks; every chunk is a `BEGIN IMMEDIATE`
tokenize+write transaction on the shared CharactersRAGDB (`DB/ChaChaNotes_DB.py:16173`).
Every UI write is also `BEGIN IMMEDIATE` (15 s busy timeout), and a chunk commit kills any
concurrent DEFERRED read-then-write writer instantly (snapshot-upgrade SQLITE_BUSY bypasses
busy_timeout — the recorded wave-1 lesson from task-21100's own review). On the first boot
after upgrading across v46 the loop runs to completion over the user's entire message
history, concurrently with the screen pre-importer, the subscriptions FTS backfill, and
actor-pack recovery (boot workers went 4 -> 7 since `35d4bf3a1`). This is the single most
plausible mechanism behind "the app recently got slower": every upgrading user gets a
first session that contends with a whole-history rebuild. The backfill itself is the fix
for a worse defect (the v46 inline freeze) — the residue is the unpaced contention window.

## Acceptance Criteria

- [x] Inter-chunk pacing exists (sleep/yield and/or contention-aware backoff) and its effect is measured: a UI write issued mid-backfill completes within a stated bound in a probe that runs a concurrent writer against an in-flight backfill
- [x] Total backfill duration for a large history is measured before/after and reported in the notes (pacing may lengthen it — state the trade explicitly)
- [x] Resumability is preserved: killing the process mid-backfill still leaves a consistent, resumable index (existing state = messages_fts_docsize membership)
- [x] The every-boot no-op probe stays cheap (one indexed scan)

## Implementation Plan

1. Add pacing to the driver loop in `DB/chachanotes_fts_backfill.py` (the one place the
   tight loop lives — `CharactersRAGDB.backfill_messages_fts` itself stays a
   single-chunk primitive):
   - A fixed inter-chunk pause (`time.sleep`, default 0.1 s) after every chunk that
     indexed rows. The no-op boot probe (first chunk finds nothing) never sleeps, so
     the every-boot cost stays one indexed scan.
   - Contention-aware backoff: a chunk that dies with SQLite's plain lock-queue
     timeout (`database is locked` from the chunk's own `BEGIN IMMEDIATE` after the
     15 s busy handler expires — the retryable kind, NOT the non-retryable
     snapshot-upgrade form, which only DEFERRED read-then-write transactions can hit)
     is retried a bounded number of times with escalating sleeps instead of killing
     the whole run until next boot. Non-lock errors keep the existing
     fail-fast-and-wrap behavior.
   - A `should_abort` callback checked between chunks; when provided, every sleep is
     sliced into <=50 ms increments that re-check it, so an in-flight sleep is
     interruptible. `app.py`'s worker passes the Textual worker's `is_cancelled` so
     app shutdown cuts the pacing instead of waiting out the paced run.
2. TDD: new `Tests/DB/test_chachanotes_fts_backfill_pacing.py` — deterministic
   pacing/backoff/abort tests written red-first against the unpaced driver, plus the
   AC's behavioural probe: a concurrent `add_message` writer against an in-flight
   backfill thread, asserting each write lands within a stated bound while the
   backfill is provably still running.
3. Measure on a synthetic history (scratchpad script, production driver, same code
   both arms with pause=0 as the "before"): (a) concurrent-writer `add_message`
   latency mid-backfill, (b) total backfill duration. Report both and state the
   pacing-lengthens-total-time trade.
4. Mutation-test the pacing line (remove the sleep, watch the pacing test go red).
5. Targeted tests teed to a file, `./scripts/preflight.sh`, tick ACs, notes, Done.

## Implementation Notes

**Approach.** Pacing lives entirely in the driver
(`DB/chachanotes_fts_backfill.py`); `CharactersRAGDB.backfill_messages_fts`
stays a single-chunk primitive. Per the owner's durable-over-clever ruling the
core is a fixed sleep, not an adaptive scheme:

- `pause_seconds` (default `INTER_CHUNK_PAUSE_SECONDS = 0.1`) after every
  chunk that indexed rows. The completion/no-op path never sleeps, so the
  every-boot probe stays exactly one indexed scan (pinned by
  `test_the_no_op_boot_probe_never_sleeps`).
- Contention-aware backoff: a chunk that dies on SQLite's *plain lock-queue*
  timeout (`database is locked` after the chunk's own 15 s busy handler — the
  retryable kind; the non-retryable snapshot-upgrade form only exists for
  DEFERRED read-then-write transactions, which the `BEGIN IMMEDIATE` chunk is
  not) is retried through a bounded schedule (`0.5/1/2 s`, counter reset on
  any successful chunk) instead of killing the run until next boot. Any other
  error keeps task-21100's fail-fast-and-wrap contract.
- `should_abort` seam, polled between chunks and inside every sleep; `app.py`'s
  worker passes the Textual worker's `is_cancelled`.

**Measurements** (synthetic history: 50,000 messages x ~2 KB ≈ 100 MB text,
NVMe, template DB copied per arm, arms interleaved x2; foreground
`add_message` every 50 ms for the whole window — the real UI write shape,
`BEGIN IMMEDIATE` on a shared instance from a second thread):

| arm | backfill total | write-lock duty cycle | fg add_message mean / median / p95 / max |
|---|---|---|---|
| before (pause=0, the old tight loop) | 1.15–1.17 s | ~100% | 153–197 ms / 111–207 ms / ~207 ms / **462–470 ms** |
| after (pause=0.1) | 11.51–11.53 s | 10% | 3.5 ms / 1.0 ms / ~11 ms / **23.5–24.5 ms** |

Chunks: 101 per run, mean 11.5 ms, max 28 ms — so the paced foreground worst
case is exactly "one in-flight chunk", while the unpaced convoy made every
write queue for hundreds of ms. **The trade, explicitly: total backfill time
grows ~10x (1.2 s -> 11.5 s here) because the run is 90% sleep.** Nobody
waits on the backfill (background thread; search fills in progressively by
design), and on the slower disks/bigger histories where the unpaced window
gets long enough to be felt, its stall ceiling grows with chunk cost while
the paced ceiling stays one-chunk-bounded. Honest calibration of the
finding's premise: on this machine the unpaced window for 50k messages is
~1.2 s, not session-long — the "whole first session" form needs a much larger
history or slower storage; what is unconditionally true is the 100% duty
cycle and the ~half-second stalls while it lasts.

**Stated bound** in the probe
(`test_ui_write_latency_stays_bounded_while_a_backfill_is_in_flight`): every
concurrent write completes in < 2.0 s (CI-generous; measured max 24.5 ms at
50k scale). The probe asserts the backfill thread is still alive after the
last write, so it cannot pass vacuously against a finished run — under the
pacing mutation the probe itself failed on exactly that guard.

**Mutation tests** (both restored): (1) pacing sleep disabled -> 3 tests red
(`test_backfill_sleeps_the_configured_pause_between_chunks`,
`test_abort_cuts_an_in_flight_pause_at_the_poll_slice`, and the latency
probe via its vacuousness guard); (2) backoff branch disabled -> both locked-
retry tests red.

**Shutdown path.** Chosen mechanism: interruptible sleeps, not
"short enough to matter" (the backoff can be 2 s). All sleeps route through
`_interruptible_sleep`; with `should_abort` present the wait is sliced into
<= 50 ms (`_ABORT_POLL_SECONDS`) abort-checked steps, so a cancelled worker
stops within one slice + one chunk. Stopping is clean, not a failure: the
frontier is `messages_fts_docsize` membership in the DB, and the next boot's
run resumes it (`test_abort_between_chunks_leaves_the_resumable_frontier`).
The chunk itself (mean 11.5 ms of work; up to the 15 s busy timeout if queued
behind a foreground writer) remains non-interruptible — pre-existing and
bounded. Resumability semantics are untouched: each chunk still commits in
its own IMMEDIATE transaction and every sleep happens outside any
transaction; task-21100's SIGKILL witness pins the kill-safe form.

**Test evidence.** New `Tests/DB/test_chachanotes_fts_backfill_pacing.py`:
8 passed (with `test_chachanotes_v49_messages_fts_update_scope.py`: 27
passed). `./scripts/preflight.sh` green after a reviewed regen of the
diagnostic inventory (+4 rows in the driver: three abort log.info + one
backoff log.warning; each interpolates only integer counts/fixed constants).
**Pre-existing dev red, NOT from this change** (baselined in a clean worktree
at `983aa5878`): 12 migration tests fail on dev itself because production
`add_message` now writes `assistant_generation_state` (a v48 column) and
breaks every `chachanotes_db_at_version(..., 44/45)` fixture that seeds
through it — 8/14 in `test_chachanotes_v47_messages_fts_backfill.py`, 4/7 in
`test_chachanotes_sync_log_retention_migration.py`. Filed as **task-22280**;
this task's new tests avoid the broken path by seeding at current schema and
issuing the v46 migration's own `'delete-all'` to reproduce the window state.

**Files.** `tldw_chatbook/DB/chachanotes_fts_backfill.py` (pacing/backoff/
abort), `tldw_chatbook/app.py` (`_backfill_chachanotes_messages_fts` passes
`is_cancelled` as `should_abort`),
`Tests/DB/test_chachanotes_fts_backfill_pacing.py` (new),
`Docs/security/production-diagnostic-inventory.json` (reviewed regen),
`backlog/tasks/task-22280 - ...md` (dev-red finding).

**Review correction (controller-recorded):** the description's sentence "Every UI write is
also BEGIN IMMEDIATE" is refuted — `add_conversation` (`ChaChaNotes_DB.py:8886`) is a plain
DEFERRED writer and dies un-retried with the snapshot-upgrade `database is locked` the
instant a backfill chunk commits (review-reproduced 3/3; `add_message` IMMEDIATE survived
10/10). Pacing shrinks that collision window ~10× but cannot close it; the residual fix
(`immediate=True` in `add_conversation` + a sweep for sibling deferred writers) is filed as
a follow-up from this burn-down.
