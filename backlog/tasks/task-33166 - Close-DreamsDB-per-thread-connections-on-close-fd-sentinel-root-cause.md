---
id: TASK-33166
title: Close DreamsDB per-thread connections on close - fd-sentinel root cause
status: Done
assignee: []
created_date: '2026-09-29 19:02'
labels:
  - dreams
  - tech-debt
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The test-suite fd-growth sentinel (Tests/conftest.py, warn-only, threshold 200) has been nudged over by Dreams test fixtures across two PRs (#2817 measured 197/200 at HEAD; #2890 fixtures pushed runs to ~295). Root cause is not the fixtures: DreamsDB.close() closes only the CALLING thread connection while the thread-local idiom keeps one connection per worker thread that ever touched the DB, and each connection also pins private_sqlite admission state. Fix close() to sweep ALL thread-local connections (registry of weakrefs or equivalent), keeping the Library_Collections_DB idiom intact; re-baseline the sentinel afterwards. Filed from the Dreams Phase 2 final-review deferred triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DreamsDB.close() releases every thread-local connection it created, verified by a per-connection closed-state probe across owning threads (fd-count asserts are CI-flaky; the fd delta is separately evidenced in the notes),fd-growth sentinel returns below threshold on the Dreams suites without loosening it,library_Collections_DB-pattern behavior (thread-local reads) unchanged
<!-- AC:END -->

## Implementation Plan (the how)

1. Whitebox N-thread close-sweep test first (TDD): 4 worker threads each force a held connection via `DreamsDB._held_connection()`, main thread calls `close()`, each owning thread probes its captured connection for `sqlite3.ProgrammingError` (closed-ness probed from the owning thread because a cross-thread execute is not a valid signal while connections are thread-bound; fd-count assertions are flaky on CI). Also asserts: registry cleared, distinct per-thread connections, second `close()` idempotent, and transparent rebuild on the next use.
2. Registry: per-thread slot becomes a small `_HeldConnectionState` holder (mutable `conn` attribute) stored on the existing `threading.local`; a `weakref.WeakSet` of holders (`DreamsDB._conn_states`) is (re-)registered on every `_held_connection()` call so a holder revived after a registry-clearing close is tracked again. Weakrefs mean a dead thread's holder (and connection) drops out automatically — same lifecycle as today.
3. `close()` sweeps every holder: clear the slot, guarded `conn.close()`, clear the registry at the end; idempotent.
4. Cross-thread close enabler: sqlite3 refuses ANY cross-thread call on a default connection — `close()` included (verified by probe: raises `ProgrammingError` and leaves the handle open) — so `_get_connection` now opens with `check_same_thread=False`, mirroring the ChaChaNotes held-connection precedent ("Required for threading.local approach"). Usage discipline stays per-thread.
5. Gate: `Tests/Dreams/` green (77 baseline + new test); fd-sentinel delta measured before/after with `TLDW_TEST_FD_GROWTH_LIMIT=1` to expose the actual number under the default-200 threshold.

ADR required: no
ADR path: N/A
Reason: bug fix inside an existing schema/idiom; no storage, interface, or policy decision changes.

## Implementation Notes (imagine this is the PR description)

- Approach: each thread's held-connection slot is a `_HeldConnectionState` holder object, strongly referenced by the thread's `threading.local` storage and weakly tracked in `DreamsDB._conn_states` (a `WeakSet`, initialized before `super().__init__` because schema init already opens the main-thread connection). `threading.local` attributes cannot be cleared from another thread; the holder makes the slot clearable by the sweeping thread, which is what preserves the existing transparent-rebuild semantics for every thread (slot reads None -> `_held_connection()` opens a fresh connection and re-registers the holder).
- `close()` now closes every held connection across all threads (guarded per connection, registry cleared at the end, idempotent), instead of only the calling thread's. No production caller closes DreamsDB today (app wiring keeps it for app lifetime); callers today are test fixtures calling close in teardown, and close was already non-terminal for the calling thread — that revive-on-next-use semantics is preserved and extended to all threads.
- `check_same_thread=False` added in `DreamsDB._get_connection` (now calling `connect_private_sqlite("db.base", ...)` directly instead of through `BaseDB._get_connection`, same owner/row_factory). Necessary enabler: sqlite3 raises `ProgrammingError` on any cross-thread call including `close()` and leaves the handle open, so the sweep could not release worker connections without it. Per-thread usage discipline is unchanged; same precedent as ChaChaNotes_DB's held connections.
- Test: `Tests/Dreams/test_dreams_db.py::test_close_sweeps_every_thread_local_connection` (4 threads; closed-ness probed from each owning thread; asserts registry-empty, distinct connections, idempotent close, transparent rebuild). Written first and confirmed failing (red) before the fix.
- Evidence: gate `pytest Tests/Dreams/ -q` = 78 passed (77 baseline + 1). fd-sentinel (session-scoped, warn-only): pre-fix Dreams-suite session grew fds by 138 (start=14, end=152, measured with `TLDW_TEST_FD_GROWTH_LIMIT=1`); post-fix the same session grows by 3 (start=14, end=17) — DreamsDB's share eliminated; the residual +3 is unrelated fixture noise. Default threshold (200) untouched.
- Observed but untouched: `Library_Collections_DB.close()` has the same close-only-the-calling-thread shape (line ~682) and leaks worker-thread connections the same way. Out of scope for this task (its close also participates in `_core_closing` maintenance gating, so a sweep there needs its own design); flagged here for a future task.
- Ruff on modified files returns to the pre-existing baseline (7 findings on HEAD; none introduced).
- Modified files: `tldw_chatbook/DB/Dreams_DB.py`, `Tests/Dreams/test_dreams_db.py`.
