---
id: TASK-21101
title: >-
  NotesDeviceStateStore - adopt the WAL+NORMAL held-connection template and stop re-running the schema census per transaction
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
labels:
  - performance
  - database
  - notes-sync
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21101).

`Notes/notes_device_state_store.py:443-472` is the app's only remaining DELETE-journal +
synchronous=FULL store (live pragma read-back confirmed: journal_mode=delete, synchronous=2).
It opens a fresh connection per operation, and `initialize_notes_device_schema` re-runs a full
`sqlite_schema` census plus 16 `CREATE INDEX IF NOT EXISTS` re-executions (~60 statements)
inside EVERY transaction, including pure reads. It sits behind the notes-sync runtime (boots
unconditionally) and the notes import executor (2-4 receipt transactions per imported note: a
500-note import pays 1,000+ open/census/fsync cycles). This store dodged the task-15465/15466
sweep because it lives outside `DB/`.

## Acceptance Criteria

- [x] The store uses the sanctioned template: held thread-local connection, WAL journal_mode, synchronous=NORMAL, isolation_level=None (exemplar `Library_Ingest_Jobs_DB.py`), with pragmas verified by read-back in a test
- [x] `initialize_notes_device_schema` runs once per connection lifetime (the `initialize()` seam), not per transaction; a statement-count probe demonstrates the reduction
- [x] Write transactions keep their current BEGIN IMMEDIATE semantics; existing notes-sync and import tests stay green

## Implementation Plan

1. Baseline: run the existing notes-device-store / notes-sync / receipts / private-sqlite
   interop test files on the base commit and record counts (tee to `test-logs/`).
2. Red-first tests in `Tests/Notes/test_notes_device_state_store.py`:
   pragma read-back on the LIVE held connection (journal_mode=wal, synchronous=1,
   foreign_keys=1, isolation_level None, persistent wal in the file); a census-call-count
   probe (monkeypatch-count `initialize_notes_device_schema`: exactly 1 for
   initialize+N operations on one thread) plus a `set_trace_callback` statement-count
   probe on a pure read (BEGIN/SELECT/COMMIT, no sqlite_schema census, no CREATE INDEX);
   per-thread connection affinity (same thread reuses one connection, another thread gets
   its own, cross-thread visibility); `close()` closes held connections and the store
   re-arms on next use.
3. Implement in `Notes/notes_device_state_store.py`:
   - `_get_connection()`: thread-local held connection created via the `_connect` seam
     with `check_same_thread=False, isolation_level=None`; on open: `foreign_keys=ON`,
     then the schema pass inside its own `BEGIN IMMEDIATE` transaction, and only after
     it succeeds `journal_mode=WAL` + `synchronous=NORMAL` (never adopt WAL on a
     database the census refuses — preserves the bytes-unchanged refusal tests).
   - `transaction()`: reuse the held connection; keep explicit `BEGIN`/`BEGIN IMMEDIATE`,
     commit on success, rollback on BaseException (no per-transaction schema pass).
   - `initialize()`: first call rides the connection-open schema pass; a repeated call
     re-validates explicitly in one immediate transaction (tamper-detection contract of
     the existing reject-unexpected-objects tests). Error mapping unchanged.
   - `close()`: best-effort close of every held connection across threads; store re-arms.
   - `_connect` stays the raw one-shot factory (read-only planning path + interop test).
   - Wire `store.close()` into `NotesSyncRuntimeOwner._shutdown_once` (existing seam).
4. Adapt `test_root_and_child_lifecycle_propagation_rolls_back_together` to the held
   connection (monkeypatched `_connect` + `store.close()` to force the reconnect through
   the seam) while preserving its rollback-atomicity semantics.
5. Run: new tests, every test file touching notes_device_state_store / notes_sync /
   note_import_receipts / private_sqlite interop+inventory, then a full
   `--collect-only` sweep. Timed before/after probe from an isolated tmp path.

## Implementation Notes

The store now holds ONE long-lived connection per thread (`threading.local`;
`check_same_thread=False`, `isolation_level=None` — the exemplar's known omission
included) instead of opening a fresh connection per operation, and the schema
census runs once per connection lifetime, at open, not inside every transaction.

Key decisions:

- **WAL is adopted only AFTER the census succeeds.** `journal_mode=WAL` is
  persisted in the database file; this store's contract is that a refused
  foreign/tampered database stays byte-identical (several existing tests assert
  `read_bytes()` equality after a refused `initialize()`). `_open_schema_ready_connection`
  therefore runs `foreign_keys=ON` → `BEGIN IMMEDIATE` + census + commit →
  only then `journal_mode=WAL` + `synchronous=NORMAL`. A new test pins this
  (`test_refused_foreign_database_is_never_switched_to_wal`).
- **`initialize()` keeps its tamper-detection contract.** A fresh store's first
  `initialize()` rides the connection-open census (exactly one pass); a repeated
  `initialize()` re-runs the census on the held connection so the existing
  reject-unexpected-objects tests still detect external tampering. Error
  mapping (Unsupported / incompatible / generic) unchanged.
- **`transaction()` semantics preserved**: explicit `BEGIN` / `BEGIN IMMEDIATE`,
  commit on success; rollback widened from `except Exception` to
  `except BaseException` because a held connection (unlike the old
  close-per-op form) must never carry a poisoned open transaction forward.
- **`close()`** closes every held connection (tracked in a lock-guarded list so
  cross-thread close works) and re-arms: the next use transparently reopens.
  Wired into `NotesSyncRuntimeOwner._shutdown_once` via the same getattr-guarded
  pattern the adapter close already uses (tests supply fake stores).
- **`_connect` stays the raw one-shot seam** (read-only planning path in
  `note_import_receipts.py` and the private-sqlite interop test pin its
  signature/owner-id); it now forwards extra kwargs so the held path opens
  through the same audited seam (`sqlite-private-owner-inventory.md` symbol
  stays accurate).

Test-rig adaptations (semantics preserved): the store rollback-atomicity test and
three receipts query-shape probes monkeypatched `_connect` per operation; they now
pass `**kwargs` through and call `store.close()` after patching so the next
operation reconnects through the traced/authorized seam.

Files: `tldw_chatbook/Notes/notes_device_state_store.py` (template + docstring),
`tldw_chatbook/Notes/notes_sync_runtime.py` (+9 lines: store close at shutdown),
`Tests/Notes/test_notes_device_state_store.py` (5 new tests + 1 rig adaptation),
`Tests/Notes/test_note_import_receipts.py` (3 rig adaptations).

Evidence (logs in `test-logs/task-21101-*.log`):

- Baseline (base `0f9638cef`, 9 touching files): 450 passed / 6 failed — all 6
  the pre-existing Actor_Packs `execute_query` AttributeError dev red
  (ProductionApp lifecycle; perf-review finding 21106).
- New tests red-first on base: 4 failed / 1 passed (the WAL-refusal guard is a
  guard on the new ordering and passes trivially on base).
- After: same 9 files → 452 passed / 6 failed (identical pre-existing set;
  diff of FAILED lines shows zero additions). Wide set (all `Tests/Notes` +
  private-sqlite + Library lasting-sync + lifecycle): 3166 passed / 6 failed
  (same set). UI sync files: 22 failures A/B-proven identical on base
  (Actor_Packs family + one unrelated SimpleNamespace harness gap);
  `test_library_shell` spot A/B also failed identically on base.
- Pragma read-back (live held connection): journal_mode=wal, synchronous=1,
  foreign_keys=1, isolation_level None; persistent wal confirmed via an
  independent connection.
- Census elimination: `initialize_notes_device_schema` called exactly once for
  initialize + 4 subsequent operations (was once per transaction); a pure
  `get_root` now executes 3 statements (BEGIN/SELECT/COMMIT) with no
  sqlite_schema census and no CREATE INDEX re-execution (was ~60).
- Timed probe (isolated tmp path): 500 `update_root_status` calls =
  1,000 transactions: 1.467s before → 0.016s after (~92x); the receipt-shaped
  workload a 500-note import pays.
- Full collect sweep: 55,013 tests collected; 33 collection errors, all
  pre-existing optional-dep/app-construction families (numpy, mlx, Confluence,
  settings sweep), none in Notes. `ruff check` clean on all four files.

Deliberately NOT touched (task-21129): `notes_sync_executor.py` loop-side call
patterns and missing indexes.
