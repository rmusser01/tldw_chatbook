---
id: TASK-23027
title: >-
  Notes sync re-reads every file and re-selects every note on every sync
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - performance
  - notes-sync
  - database
priority: medium
---

## Description

Three compounding inefficiencies in the notes-sync path, all pre-existing and in neither prior review:

1. **`observe_root` re-reads everything per sync.** At N=1000 with **nothing changed**: 350-370 ms,
   1,000 SELECTs, 1,017 file opens, worst event-loop stall 36-48 ms. It already computes a
   `_discovery_signature` (`notes_sync_runtime.py:651`) that is never used to skip. It runs at boot.
2. **`to_thread(lambda: asyncio.run(coro))` 9 times per synced note** - 684 us each against 0.1 us
   for a plain `await`; ~6-8 ms of the 16.5 ms per note, across 22 sites.
3. **~1 abandoned SQLite connection per synced note** - at N=1000, 1,010 opened, 0 closed.

## Acceptance Criteria

- [x] A sync with no filesystem changes does not re-read every file or re-select every note - use the signature that is already computed
- [x] The nested `asyncio.run`-in-a-thread pattern is removed from the per-note path
- [x] Connections opened per sync are bounded; measured before and after at N=1000
- [x] Worst event-loop stall during a no-op sync is measured and reported
- [x] Sync correctness is unchanged - this area has a data-loss history, so a faster path that risks a lost or mis-ordered write is not acceptable

## Evidence

Counted live at N=1000. The loop-stall figures come from a probe with no rendering competing, so they
are a **floor**, not a ceiling.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Plan

1. Reproduce all three findings on this base with an isolated-config probe (temp DB, temp root,
   N=1000): wall, per-note SELECTs, file opens, sqlite connects, worst loop stall — interleaved arms.
2. Fix 1 (skip the re-read): per-item observation reuse inside `_ProductionRuntimeAdapter.observe_root`,
   validated by the SAME metadata the existing watcher signature trusts — a cached file snapshot is
   reused only when the current discovery stat equals the snapshot's own open-time `reviewed_state`
   (device, inode, size, mtime_ns, ctime_ns); a cached note snapshot is reused only when one bulk
   (id → version, deleted) read confirms the exact version the snapshot carries. Bindings and root
   state are always re-read (they are cheap and they gate `bound`), so no signature has to cover them.
   Prove per mutation class — file edited / added / deleted / renamed / mtime-only touch / DB-side
   note edit / DB-side note delete — that reuse does NOT happen and observations equal a cold pass.
3. Fix 2 (kill `to_thread(lambda: asyncio.run(...))`): one persistent event loop per worker thread
   (module helper in `notes_sync_executor.py`), used by all 22 executor sites and the runtime's
   `observe_notes`. Thread placement, `_joined_thread_call` shielding, and interleaving semantics
   are unchanged — only the per-call loop construction goes away.
4. Fix 3 (bound connections): measure first. The abandoned connections are expected to be the
   per-`asyncio.run` throwaway inner-loop executor threads meeting thread-local DB handles; loop
   reuse should bound them. If the measurement disagrees, apply the held-connection shape instead.
5. Tests: mutation-class coverage for reuse, cold/warm equivalence, skip-then-real-change walk,
   quit-with-sync-in-flight at the new reuse points, error-mid-sync leaves store usable, and
   executor-path interleaving (concurrent writer vs reused observation). Every new test run against
   a deliberately broken implementation; per-test mutation results recorded.
6. Re-measure at N=1000 interleaved; run the Notes suites A/B against base for reds; preflight.

## Implementation Notes

All three legs of the finding reproduced on base `c4e52794e2` before any change (isolated-config
probe, N=1000, nothing changed: 356-396 ms, 1,000 per-note SELECTs, 1,000 file opens, worst loop
stall 26.5-28.3 ms; executor path: 8.0 `asyncio.run` per synced note, 1,004 connections opened per
1,000-note sync).

**Fix 1 - per-item observation reuse, not a whole-pass signature skip.** The filing suggested using
`_discovery_signature` to skip the pass wholesale. That shape was rejected after reading the code:
the stored `_root_signatures` map is OVERWRITTEN by the watcher's `changed_root_ids` before the
hinted reconcile runs (so a signature-keyed skip would skip the very pass the watcher just asked
for -- a data-miss bug), and the signature covers only the filesystem, never DB-side note edits or
binding transitions. Instead `_ProductionRuntimeAdapter` now keeps one `_ObservationReuse` cache
per root, and `observe_root` revalidates every entry per pass against authoritative state read
fresh THIS pass: a file snapshot is reused only when the current discovery stat (device, inode,
size, mtime_ns, ctime_ns -- the same metadata the shipped watcher signature already trusts) equals
the stat captured at the moment the cached bytes were read; a note snapshot is reused only when
one bulk `observe_versions` read confirms the exact version it carries (every notes write path
bumps `version` under optimistic locking). Bindings and root state are never cached --
`list_bindings` runs fresh every pass -- so binding/root transitions need no signature coverage at
all. The cache is content-bounded (32 MiB/root; over-budget items are simply re-read), rebuilt
off-loop only on success, and dropped on `close()`. The boot reconcile's FIRST pass stays a full
cold read by design: trusting persisted state to skip reads at boot would miss offline edits, and
sync correctness outranks the win. Every later no-change pass (sync-now, check, watcher hint,
post-apply verification) is the one that stops paying.

**Fix 2 - `run_worker_coroutine` replaces `to_thread(lambda: asyncio.run(...))`.** The authority
seam is async-shaped but its work is synchronous (or internally thread-offloaded), so those
coroutines must be finished on the worker thread. The new module helper keeps ONE event loop per
worker thread (`threading.local`) instead of building and tearing down a loop -- and, on the folder-
repository seams, a throwaway default-executor thread -- per call. All 22 executor sites plus the
runtime's note-observation batch now use it. Thread placement, `_joined_thread_call`'s
finish-before-redelivering-cancellation shielding, and interleaving semantics are byte-identical;
only the per-call loop/executor/connection churn disappears. An AST ratchet test keeps
`asyncio.run` out of the executor.

**Fix 3 - connections: hold, not close.** Measured first, per the WAL-anchor counter-lesson: the
~1 abandoned connection per synced note was the per-`asyncio.run` throwaway inner-executor thread
meeting `CharactersRAGDB`'s thread-local connection (fresh thread -> fresh connection -> thread
dies with `shutdown_default_executor`). No `close()` was added anywhere -- loop reuse alone bounds
connections to pool width: 1,004 -> 3 per 1,000-note sync (the 3 are one-time pool-thread
warmups), threads started 2,003 -> 3.

**Measured (interleaved A/B in one process, quiet machine; A = pre-change path, B = shipped):**

- observe_root, N=1000, nothing changed: wall 392 -> 84 ms median; per-note SELECTs 1,000 -> 0
  (one bulk version SELECT replaces them); file opens 1,000 -> 0; worst loop stall 27.3 -> 26.0 ms
  (the stall is the pre-existing loop-side build/plan tail, present in both arms; an earlier
  apparent stall regression to ~80-100 ms was disproved by interleaving -- it was concurrent
  machine load). Under heavy concurrent load the same interleaved pair read 1,332 -> 165 ms.
- executor per synced note (UPDATE_NOTE, K=20 x3 interleaved): 13.1 -> 6.6 ms/note; connections
  1.10 -> 0.15/note; threads 2.10 -> 0.15/note. At K=1000: connections 1,004 -> 3.

**Signature-coverage proof per mutation class** (each asserts the warm pass re-reads the changed
item AND equals a cold adapter's ground-truth pass): file edited; file edited same-length (only
mtime/ctime can catch it); file added; file deleted; file renamed (same inode, new path); mtime-
only touch (same bytes -- must re-read); DB-side note edit; DB-side note delete; new binding
between passes. Walks: skip-then-real-change never stays skipped (and re-warms over the NEW
content); failed pass leaves cache and store usable; bulk-read failure propagates instead of
serving stale; cancel mid-pass (quit with sync in flight) leaves the adapter usable and the next
pass correct; concurrent writer x10 warm passes yields only committed (version, content) pairs and
settles to cold ground truth.

**Mutation results: 15 deliberate single-point defects, 15 detected** (restores Edit-based and
re-verified green): stat compare drops mtime/ctime (4 reds); stat compare always-true (6); cached
note served without version check (8); note cache keyed by binding_id -- wrong identity space (3,
incl. the warm zero-selects test); cache never rebuilt after first pass (1: skip-then-change);
helper builds a new loop per call (1); helper swallows exceptions (7); `asyncio.run` reintroduced
at one executor site (1: AST ratchet); authority includes tombstones (1 -- at its own level only,
because version-bump-on-delete is a second line of defence, tested per the TASK-21127 lesson); DB
projection drops tombstones (1); chunk loop truncated at 500 (1); service ignores the bulk
projection (2); reuse budget ignored (1); `close()` keeps the cache (1); bulk-read failure
swallowed into stale reuse (1).

**Files:** `tldw_chatbook/Notes/notes_sync_runtime.py` (reuse cache + revalidation, worker-loop
helper use, off-loop cache rebuild, periodic yields on the awaitless hit path),
`tldw_chatbook/Notes/notes_sync_executor.py` (`run_worker_coroutine` + 22 site swaps),
`tldw_chatbook/Notes/notes_sync_authority.py` (`observe_versions`),
`tldw_chatbook/Notes/notes_scope_service.py` (`get_note_version_states_for_sync`, with a per-note
fallback for backends without the projection), `tldw_chatbook/Notes/Notes_Library.py` and
`tldw_chatbook/DB/ChaChaNotes_DB.py` (`get_note_version_states`: chunked-at-500, one read
transaction, tombstones included), plus new tests
`Tests/Notes/test_notes_sync_observation_reuse.py` (18),
`Tests/Notes/test_notes_sync_worker_coroutine.py` (7),
`Tests/Notes/test_notes_sync_version_states.py` (9).
