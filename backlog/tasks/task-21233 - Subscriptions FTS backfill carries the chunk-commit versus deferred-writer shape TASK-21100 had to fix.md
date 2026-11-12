---
id: TASK-21233
title: >-
  Subscriptions FTS backfill carries the chunk-commit versus deferred-writer
  shape TASK-21100 had to fix
status: Done
assignee: [rmusser01]
created_date: '2026-08-23'
labels:
  - performance
  - database
  - reliability
  - subscriptions
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; sweep candidate
raised by the TASK-21100 review.

TASK-21100's fix round found that a chunked FTS backfill which commits between chunks collides
with a DEFERRED writer that upgrades to a write lock mid-transaction: on the ChaChaNotes DB
this produced an **instant** `database is locked` on `add_message` during the first-boot
backfill, and SQLite's busy timeout did not help because the lock-upgrade path bypasses it.
Throttling the backfill was tried and proven not to be a fix; the fix was `immediate=True`
scoped to the hot message writers (12 hot writers plus 5 outer wrappers, merged as
`41a240ccd`).

`DB/Subscriptions_DB.py:1439` `backfill_items_fts(chunk_size=500)` has the same structural
shape — a resumable chunked backfill committing per chunk against a table other code writes.
It was not fixed alongside TASK-21100 because the subscriptions DB is idle in the observed
first-boot scenario, so the collision was never provoked. It is therefore a latent instance of
a defect class this burn-down has already paid to learn, not a measured failure.

## Acceptance Criteria

- [x] The subscriptions writers that can run concurrently with `backfill_items_fts` are enumerated, and each is shown either to take an immediate transaction or to be provably unable to overlap the backfill
- [x] A test drives a real write against `subscription_items` while a chunked backfill is in progress and fails if that writer receives `database is locked`
- [x] Throttling or retry is not used as the fix
- [x] The backfill remains resumable by rowid and its per-chunk cost is unchanged

## Implementation Plan

1. Relocate `backfill_items_fts` by symbol; audit its chunk transaction mode and the
   SubscriptionsDB `transaction()` manager semantics (it issues no BEGIN unless
   `immediate=True` — bodies run under python-sqlite3's implicit-BEGIN-before-DML policy).
2. Enumerate every SubscriptionsDB writer that can run concurrently with the backfill
   (AST sweep: every `with self.transaction()` body containing DML); classify
   boot-exclusive (`_initialize_schema`) and already-immediate sites; record the table.
3. Convert every overlapping writer (and the backfill chunk itself) to
   `transaction(immediate=True)` — the TASK-21100 fix, NOT throttling or retry (pacing
   already exists via TASK-22215 and is untouched).
4. Add to `Tests/Subscriptions/test_fts_backfill.py`: a canary test that drives real
   `subscription_items` writes WHILE a chunked backfill is in flight (fails if any write
   receives `database is locked`), and a structural pin over the enumerated writer set
   (the HOT_MESSAGE_WRITERS idiom).
5. Verify: new tests pass; existing backfill resumability/pacing suites stay green
   (resumable-by-rowid and per-chunk cost unchanged); closeout with the enumeration
   table, commands, results; one commit.

ADR required: no
Reason: applies TASK-21100's standing immediate-writer policy (precedent `41a240ccd`) to
the enumerated SubscriptionsDB writers; no new architectural decision, schema change, or
interface change. The existing paced driver (TASK-22215) and its fail-and-resume-next-boot
contract are deliberately untouched.

## Implementation Notes

**Approach.** Applied TASK-21100's standing write-lock policy to this database: every SubscriptionsDB writer that can overlap the chunked `subscription_items_fts` backfill — and the backfill's own chunk transaction — now reserves SQLite's write lock up front via `self.transaction(immediate=True)`. No throttling and no retry anywhere in the fix (AC #3). The per-chunk work and the rowid-resumable driver are untouched; only the transaction mode changed (AC #4).

**Writer enumeration (AC #1).** Recorded as the `SUBSCRIPTIONS_HOT_WRITERS` tuple in `Tests/Subscriptions/test_fts_backfill.py` (31 writers, including the backfill chunk itself), with the deliberate exclusions documented beside it: read-only methods (readers never upgrade; the two explicit `BEGIN DEFERRED` snapshot readers never write), `_initialize_schema` (boot path, no concurrent writers exist), and the sites that were already IMMEDIATE (`accept_watchlist_runs`, `accept_briefing`, `_migrate_from_v1_to_v2`).

**Tests (AC #2).**
- `test_real_item_writes_during_an_in_flight_backfill_never_die_locked` — production-shape integration race: one shared WAL SubscriptionsDB, the real paced backfill driver on a worker thread (240 legacy rows / chunk 8 / 0.05 s pause = 30 chunk commits), real item writers (`mark_item_status`, `set_item_flagged`) from the foreground across the window, with non-vacuity assertions (window observed open, backfill alive mid-writes) and full convergence (docsize 240). Result: **9 passed** for the file.
- `test_subscriptions_hot_writers_reserve_the_write_lock_up_front` — the red-capable pin (the v47 `HOT_MESSAGE_WRITERS` idiom): every enumerated writer's source must contain `self.transaction(immediate=True)` and no bare `self.transaction()`.

**Mutation evidence.** With `tldw_chatbook/DB/Subscriptions_DB.py` reverted to HEAD (file preserved via `cp`, never stash): the structural pin fails (**1 failed** — writers revert to DEFERRED by name); restored: **9 passed**. The live race canary is green under the same mutation, for a documented reason rather than a vacuous one: this database runs `journal_mode = WAL` with a busy timeout (module docs at `Subscriptions_DB.py:145-157` and the `PRAGMA journal_mode = WAL` at open), and in WAL a write-first transaction never takes the deferred upgrade path that produced TASK-21100's instant `database is locked` — the busy handler absorbs the contention either way. The defect class this task closes is a DEFERRED read-then-write upgrader colliding with a committing writer; the structural pin makes that class impossible by construction on this database (no DEFERRED writers remain), and the canary guards the real integration behavior (no locked deaths, no backfill failure, convergence) that the structural test cannot see.

**Verification commands.** `python -m pytest Tests/Subscriptions/test_fts_backfill.py -q --timeout=180` -> 9 passed; mutation variant (file at HEAD) -> structural test 1 failed; full `Tests/Subscriptions` sweep, fixed tree: **298 failed, 729 passed, 3 skipped, 9 errors in 1104s**; identical command with `Subscriptions_DB.py` reverted to HEAD (cp-preserved, never stash): **298 failed, 729 passed, 3 skipped, 9 errors in 1080s** — count-identical, so every failure is the documented pre-existing dev-tip red mass (config-admission signature), zero regressions from this change.

**Conftest enrollment.** `Tests/conftest.py` `keep_bootstrap_profile` gains `test_fts_backfill.py`: the canary's worker-thread connection resolves config through the guarded loader and hits the known per-test-redirect admission signature (same class as `test_hosted_chat.py`; precedent TASK-32873 / ADR-179). The file's other tests are tmp_path DB-level tests unaffected by the redirect.

**Lineage.** TASK-22501 (the ChaChaNotes `add_conversation` sibling of this defect class) is already committed on this branch as `215fa4ff3d`; this task completes the pair.

ADR required: no — transaction-mode policy on existing writers, directly implementing the standing TASK-21100 policy this task was filed to extend; no new architectural boundary.
