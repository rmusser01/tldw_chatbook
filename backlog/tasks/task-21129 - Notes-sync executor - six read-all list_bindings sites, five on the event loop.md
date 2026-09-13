---
id: TASK-21129
title: >-
  Notes-sync executor - six read-all list_bindings sites, five on the event loop
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - notes-sync
  - database
priority: medium
dependencies:
  - TASK-21101
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21129).

`Notes/notes_sync_executor.py:1144,1256,1761,1978,2260,2682` each read EVERY binding for the
root (notes_device_state_store.py:889-897 - no LIMIT; no index on root_id found), and five of
the six are invoked without to_thread from async methods (unlike notes_sync_runtime.py:1363
which wraps correctly) - ~3K full owner-set reads per sync batch, each also paying TASK-21101's
per-op connect + census until that lands.

## Acceptance Criteria

<!--
Two clauses of the filing were disproved by measurement before implementation and
are amended in place (originals kept below each item). See Implementation Notes,
"Where the brief was wrong".
-->

- [x] No executor site materializes every binding of a root: each reads only the
      columns it consumes, through a predicate the existing
      `idx_notes_sync_bindings_root(root_id, state, binding_id)` index serves
      *(amended: the filing asked for "indexed predicates ... an index on the binding
      root/owner columns exists". That index already existed and SQLite was already
      using it; 88% of the measured cost was Python dataclass hydration, not the
      SQL. Adding another index would have bought nothing.)*
- [x] The five reconcile-projection sites, which live in coroutines, route through
      `asyncio.to_thread`; the sixth site is the synchronous new-candidate authority
      guard and deliberately stays on the loop, made cheap instead
      *(amended: the filing said five of six ran without `to_thread`. A census of
      the executor suite found **183 of 183** binding reads ran on a thread with a
      running event loop -- all six, not five. Giving the guard an await point would
      have inserted an interleaving window between an authority check and the
      mutation it admits, which sync correctness forbids.)*
- [x] Sync outcomes unchanged - existing executor tests green
- [x] Measured before/after recorded for the loop-blocking the change removes, and
      the removed work proven gone rather than relocated

## Implementation Plan

1. Census every binding read the executor performs during its own test suite,
   recording per call whether the calling thread had a running event loop -- the
   only honest test of "this is on the loop".
2. Measure the cost of the two read shapes at 10/100/1,000/5,000 bindings and split
   it into SQL time vs Python row hydration, and read `EXPLAIN QUERY PLAN` for the
   real 13-column statement, before assuming the filing's fix is the right one.
3. Add narrow store projections for exactly what the call sites consume; keep the
   result byte-identical to the read they replace (same subset, same order).
4. Route the five coroutine-side projections through one `to_thread` helper; leave
   the synchronous authority guard synchronous.
5. Prove equivalence and thread placement with tests, then run every test against a
   deliberately broken implementation and require each to fail.
6. Walk quit-with-a-sync-in-flight, the store-read failure path, and empty-DB.

## Implementation Notes

Six sites in `NotesSyncExecutor` read every binding of a sync root through
`NotesDeviceStateStore.list_bindings` and threw most of each row away. Two narrow
store projections replace them, and the five that live in coroutines now read off
the event loop.

**What changed**

- `Notes/notes_device_state_store.py`: two new private projections.
  `active_binding_note_ids(root_id, *, exclude_binding_id=None)` returns just the
  note ids of the root's active bindings, ordered by binding id;
  `has_binding_for_note_or_path(root_id, *, note_scope_id, note_id, relative_path)`
  answers the new-candidate guard's question with a `LIMIT 1` probe. Both are served
  by the pre-existing `idx_notes_sync_bindings_root` index; the projection's
  `state = 'active'` predicate also lets that index supply the ordering, so it no
  longer builds the temporary B-tree the old statement did.
- `Notes/notes_sync_executor.py`: the five reconcile stages call one new helper,
  `_desired_managed_memberships`, which awaits `asyncio.to_thread` around the
  projection; `_require_new_candidate_owner` calls the LIMIT-1 probe and stays
  synchronous on purpose.

**Where the brief was wrong** (measured before implementing, per the review's own
"corrections found during implementation" discipline)

1. *"no index on root_id found"* -- `idx_notes_sync_bindings_root(root_id, state,
   binding_id)` has existed since the feature's first commit (`aa579990d`,
   notes_device_state_schema.py:317) and `EXPLAIN QUERY PLAN` shows the old
   statement already used it. The cost was elsewhere: at 1,000 bindings the SQL
   `fetchall` was 1.83 ms of a 14.75 ms read -- **88% was `_binding_from_row`
   hydrating thirteen columns, a nested serialization profile and an enum per row**
   for callers that kept one string. No index was added.
2. *"five of the six are invoked without to_thread"* -- the census says **six of
   six**: 183 of 183 binding reads in `Tests/Notes/test_notes_sync_executor.py` ran
   with a running event loop. The five in coroutines are now off-loop; the sixth is
   a guard whose whole value is that nothing runs between it and the mutation it
   admits, so it was made cheap rather than asynchronous.
3. *"~3·K full owner-set reads per sync batch"* -- undercounted. One CREATE_NOTE
   action performs **6** full-root reads (5 guard + 1 projection), measured by
   censusing a single test.

**Measured**

Per read, by root size (median, warm; `bench_bindings.py`):

| bindings | projection before → after | guard before → after (miss) |
|---|---|---|
| 100 | 1.475 ms → 0.039 ms | 1.444 ms → 0.028 ms |
| 1,000 | 15.31 ms → 0.336 ms | 15.88 ms → 0.250 ms |
| 5,000 | 76.73 ms → 1.887 ms | 75.77 ms → 1.215 ms |

Per action, with a 1 ms event-loop lag sampler running (`bench_action.py`, the
measured 5-guard + 1-projection mix):

| bindings | base ms/action | after ms/action | longest single loop stall |
|---|---|---|---|
| 100 | 14.76 | 0.69 | 147.7 ms → 0.72 ms |
| 1,000 | 149.61 | 2.96 | 1495 ms → 2.40 ms |
| 5,000 | 477.77 | 8.46 | 1911 ms → 6.31 ms |

Work removed, not relocated: across the executor suite the call count is unchanged
at 183, but loop-side time falls **5.81 ms → 1.91 ms** and binding *records*
materialized fall **41 → 0** (21 note-id strings and 127 booleans instead). 56 of the
183 reads now report no running loop; the remaining 127 are the deliberate guard.
`asyncio.to_thread` costs 0.039 ms per hop warm, and 2.19 ms once per worker thread
that first touches the store (connect + schema census + WAL pragmas).

**Safety walks**

- *Quit with a sync in flight*: the projection is a pure read, so cancelling at its
  new await point can never abandon a half-applied mutation --
  `test_shutdown_cancel_during_the_projection_leaves_a_resumable_operation` cancels
  while the worker thread is inside the read and asserts the thread finishes
  harmlessly, the durable operation stays resumable, and the store closes. The
  runtime already quiesces via `settle()` before `close()`
  (notes_sync_runtime.py:2797,2819), so the new awaits sit inside tasks shutdown
  already joins.
- *Error path*: a failing store read now re-raises at the await instead of inline.
  `test_projection_failure_becomes_attention_not_an_unhandled_exception` pins that
  the operation still ends as a durable NEEDS_ATTENTION result rather than escaping
  into the Textual event handler that awaited it.
- *First boot / empty DB*: a root with no bindings projects `()` and the guard
  answers False, as the retired code did; TASK-21112 means a zero-profile boot never
  opens this store at all. Drove the harder variant too -- the store's *first ever*
  touch arriving on a `to_thread` worker instead of the loop thread: the database
  file did not exist, the worker opened it, ran the census, adopted WAL, and returned
  `()` / `False`; one connection held, `close()` clean, re-arm clean.
- *Interleaving* (a thread hop can wake a dormant race): the projection feeds
  `reconcile_managed_memberships`, which REMOVES any managed membership absent from
  it, so a cross-thread read that missed a just-committed binding would silently
  unfile a live note. Two tests cover it -- one asserts read-your-own-writes across
  the hop for twelve successive commits, one runs twenty concurrent projections
  against a writer flipping a binding and requires every result to be a whole
  committed set.

**Mutation results**: 11 deliberate defects, **11 detected**, restores verified
byte-identical -- projection back on the loop; guard back to the read-all scan;
`state='active'` filter dropped; ordering reversed; `exclude_binding_id` ignored;
probe `OR`→`AND`; probe ignoring `root_id`; stale cached snapshot; torn (truncated)
result; read failure swallowed; cancellation swallowed. The `root_id` mutant survived
the first round because the "never sees another root" assertion used a note that
existed on no root at all; the fixture now gives the other root values that exist
only there.

**Tests**: `Tests/Notes` on this branch **3137 passed / 2 failed / 5 skipped**; on
pristine dev `f49956038` **3129 passed / 2 failed / 5 skipped** -- the same two
(`test_notes_library_unit.py::*::test_get_db_new_instance`, a stale
`CharactersRAGDB(...)` mock double that TASK-19900's `console_library_migration_seed`
argument outgrew), and the +8 are this task's new tests. The two adjacent UI reds
(`test_library_notes_lasting_sync_flow.py::test_activation_result_routes_*`,
`activate_root() missing 1 required positional argument`) are likewise identical on
base. `./scripts/preflight.sh`: 4 of 5 checks green; the production-diagnostic-
inventory check reports the *same* `library_screen.py` digest transition on pristine
`f49956038` (a file outside this diff), so it was NOT regenerated -- absorbing another
change's unreviewed drift is precisely what that artifact exists to prevent.

**Files**: `tldw_chatbook/Notes/notes_device_state_store.py`,
`tldw_chatbook/Notes/notes_sync_executor.py`,
`Tests/Notes/test_notes_device_state_store.py`,
`Tests/Notes/test_notes_sync_executor.py`,
`backlog/docs/lessons-testing-evidence.md`.
