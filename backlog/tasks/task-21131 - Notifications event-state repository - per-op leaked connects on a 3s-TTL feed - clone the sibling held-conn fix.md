---
id: TASK-21131
title: >-
  Notifications event-state repository - per-op leaked connects on a 3s-TTL feed - clone the sibling held-conn fix
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-24'
labels:
  - performance
  - notifications
  - database
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21131).

`Notifications/event_state_repository.py:85-106` opens per-op (GC-leaked `with conn:`; no FK on
the file branch), 3+ opens per `build_server_notification_feed` call on the Home screen's 3 s
TTL cache in server mode. The sibling `client_notifications_db.py:69-108` is already the held
thread-local template with a liveness ping - clone it.

## Acceptance Criteria

- [x] #1 The repository holds one connection per thread (template shape, with the liveness ping) and closes them explicitly; FK enabled consistently on every connection
- [x] #2 Server-mode feed behavior unchanged
- [x] #3 Every read-modify-write body is one `BEGIN IMMEDIATE` transaction (added during implementation: holding a connection per thread turns the pre-existing dedupe TOCTOU into a live race - 9 of 12 concurrent writers raise `IntegrityError` on base)
- [x] #4 The module's own private-SQLite owner entry describes the private file the app actually gives this store (added during implementation: the file branch had to move off `db.base` to pass `check_same_thread=False`, and the seam enforces module-owned owner ids; enforced target kinds are unchanged)

## Implementation Plan

1. Build the measurement harness BEFORE the fix: a probe driving the real
   `build_server_notification_feed` against a real FILE-BACKED repository,
   counting connection opens through the private seam, and separately
   timing one `connect_private_sqlite` open against the statement it is
   opened to run (the TASK-21127 reframing).
2. A/B against a pinned base worktree at `68c061984`, arms interleaved
   (TASK-21130 lesson). Decide from the numbers, not the brief.
3. Held connection per thread (`dict[thread_ident, Connection]`), liveness
   ping, `isolation_level=None`, explicit `transaction()` doing
   BEGIN IMMEDIATE / COMMIT / ROLLBACK; `:memory:` keeps its single shared
   connection (closing it destroys the database).
4. Make every read-modify-write one transaction before any concurrency can
   interleave them.
5. `close()` that re-arms and refuses a connection whose thread is
   mid-operation.
6. Mutation-verify every new guard against a deliberately broken
   implementation; walk quit / error / empty explicitly.

## Implementation Notes

`EventStateRepository` now holds ONE connection per thread for the life of the
store instead of opening (and GC-leaking) a fresh one per operation, and every
read-modify-write body is a single `BEGIN IMMEDIATE` transaction.

**Connection lifecycle.** `connection()` and `transaction()` return a held
connection from a `dict[thread_ident, _HeldConnection]` (not a
`threading.local`: `close()` has to reach connections it does not own), opened
through `connect_private_sqlite` with `check_same_thread=False` and
`isolation_level=None`, with the sibling's 30 s liveness ping. All 18
`with self._get_connection() as conn:` sites became one or the other --
sqlite3's connection context manager is a *transaction* manager, not a closer,
which is what leaked them. `:memory:` still shares exactly one connection and
keeps sqlite3's same-thread guard. `_initialize_schema` keeps its own
short-lived connection (it runs inside `_ensure_schema`'s lock, which
`_get_connection` re-enters) and still closes it. `PRAGMA foreign_keys = ON` is
now asserted on every connection, not just the schema one -- currently inert,
since this schema declares no FOREIGN KEY constraints, but it no longer drifts
if one is added.

**Atomicity is a prerequisite, not polish.** `record_event_and_advance_processed_cursor`,
`remember_event`, `mark_event_presented_and_advance_high_water`, `reset_cursor`,
`record_observer_status`, `set_retention_policy`, `prune_stream_state` and
`clear_server_profile_state` all read before they write. That was harmless only
while the store opened a fresh connection per call and the only writer was the
event loop. Proven live: 12 threads recording the same event against the base
implementation produce **9 `IntegrityError: UNIQUE constraint failed` out of
12**; after, exactly one insert and eleven duplicates, zero exceptions.
IMMEDIATE rather than deferred is mandatory: under `isolation_level=None` a
deferred begin takes a read snapshot whose later write fails `BUSY_SNAPSHOT`,
which SQLite's busy handler does not retry (`busy_timeout` reads back as 5000,
pinned by a test).

**`close()`.** Closes every held connection that is not mid-operation, leaves a
busy thread's connection alone (closing it under live work is the TASK-21101
`ProgrammingError: Cannot operate on a closed database` class), and re-arms.
For `:memory:` it also clears `_schema_ready`, because the in-memory database
dies with its connection -- on base, a `close()`-then-reuse left the store
querying tables that no longer existed.

**Owner registry.** The file branch had to stop going through
`BaseDB._get_connection` in order to pass `check_same_thread=False`, and the
seam enforces that a module only names its own registered owner
(`test_private_sqlite_seam_calls_use_literal_module_owned_ids`). The
`notifications.event_state` entry said "currently uses only an in-memory
database", which has not been true since `build_server_parity_state_repositories`
started handing it `~/…/tldw_chatbook_event_state.db`; that file was being
opened under `db.base`. The entry now says `_PRIVATE_OR_MEMORY` -- **exactly the
two target kinds `db.base` already allowed**, so the enforced boundary is
unchanged -- and the store was added to `FILE_OWNER_CASES`, which subjects it to
the same 0600 / unsafe-parent / missing-parent rejection contracts as the other
file owners (four new assertions it was not covered by before).

### Where the brief was wrong

- **"3+ opens per `build_server_notification_feed`"** -- the read path (what
  Home actually calls, `mark_presented=False`) opens exactly **2**: one for
  `list_events`, one for `get_replay_status`. With `mark_presented=True` it is
  `2 + one per row` (**22** at 20 rows). The higher-volume leak the finding did
  not mention is the observer's write path: **1 open per SSE event**.
- **"no FK on the file branch"** -- true, and now fixed, but **inert**: the
  schema declares no FOREIGN KEY constraints anywhere, so the pragma changes no
  behaviour today. It is set for uniformity, not for a bug it closes.
- **"clone the sibling"** -- the named template (`ClientNotificationsDB`) uses
  `threading.local` and cannot close another thread's connection. Cloning it
  literally would have shipped a `close()` that never reaches the
  `asyncio.to_thread` pool connections this store actually accumulates. See the
  lessons entry.
- **The loop-offload leg does not apply here.** Home already runs this feed
  through `asyncio.to_thread` (`refresh_active_work_cache_async`) when every
  seam is confirmed file-backed. Only the cold-cache compose fallback runs it
  inline. No offload work was needed or done.

### Evidence

Probes and logs in `test-logs/` (gitignored); A/B against a pinned base
worktree at `68c061984`, arms **interleaved**, 5 pairs, medians.

| | base | after | |
|---|---|---|---|
| one `connect_private_sqlite` open + pragmas | 0.554 ms | -- | |
| one `list_events` SELECT on that connection | 0.049 ms | 0.049 ms | **unchanged in both arms** |
| feed read (`mark_presented=False`), opens/build | 2 | **0** | |
| feed read, ms/build | 1.398 | **0.144** | 9.7x |
| feed + mark presented (20 rows), opens/build | 22 | **0** | |
| feed + mark presented, ms/build | 15.39 | **0.899** | 17.1x |
| observer write, opens/event | 1 | **0** | |
| observer write, ms/event | 0.896 | **0.129** | 6.9x |
| whole probe (20 writes + 35 feed builds), opens | **191** | **2** | schema + one held |
| 30 feed builds INLINE on a loop, total | 44.60 ms | **5.12 ms** | 8.7x |
| worst contiguous loop gap | 6.66 ms | **1.45 ms** | 4.6x |
| independent 1 ms ticker wakeups during that window | 10 | 3 | 0.22/ms -> 0.65/ms |

The connect was **91%** of a feed read's DB cost (0.554 of 0.603 ms). Work is
GONE, not relocated: `select_ms_median` is byte-for-byte the same in both arms
(0.0487 / 0.0494), and `feed_total` / `replay_state` are identical.

- **Mutation results: 15 of 15 go RED**, `__pycache__` cleared per mutation and
  the implementation verified byte-identical after each restore
  (`test-logs/mutations.txt`). Two initially stayed GREEN and were traced
  rather than written off:
  - *`close()` closes a busy connection anyway* stayed green because
    `check_same_thread` defaulted to True, so a cross-thread `close()` was
    raising and being swallowed -- the dict-vs-`threading.local` rationale was
    false for this store. Fixed at the mechanism (`check_same_thread=False`),
    not the assertion.
  - *deferred `BEGIN`* went green after the acquisition/registration lock hold
    was tightened, because threads that open their connection after the barrier
    never overlap. Fixed by warming each worker's connection before the barrier.
  - A third, found by mutation: `pytest.raises(sqlite3.ProgrammingError)` on a
    connection passes for a wrong-thread refusal as well as a closed handle;
    it now `match=`es `"closed database"`.
- **Red-first: 14 of the 18 new guards fail on pristine base `68c061984`**
  (whole file copied into a pinned base worktree). Headline signature:
  *"concurrent recording raised: [IntegrityError('UNIQUE constraint failed:
  event_records.dedupe_key') x9]"*. The other 4 are behaviour-preservation
  guards, each mutation-covered.
- **Quit / error / empty walk**, each with a test:
  - *quit with the feed in flight*: `close()` from another thread while a
    worker is parked mid-transaction -- the operation still completes
    (`test_close_leaves_an_in_flight_operations_connection_open`); mutation
    reproduces the exact TASK-21101 `ProgrammingError`.
  - *quit with idle worker connections*: `close()` from the main thread really
    closes a worker pool thread's connection
    (`test_close_releases_an_idle_worker_threads_connection`).
  - *error*: a failed body rolls back and leaves no open transaction; a failing
    ROLLBACK never masks the original exception; `mark_event_presented` for an
    unknown event raises `KeyError` and the store stays usable; a corrupt
    database raises `DatabaseError: file is not a database` and RETRIES its
    schema pass once the file is replaced (`_schema_ready` stays False).
  - *empty / first run*: construction still creates no file (TASK-21105
    contract, existing tests green), and a first-run feed returns
    `items=[] total=0 replay.state="empty"` having opened exactly 2 connections.
- **Tests.** `Tests/Notifications + RuntimePolicy + Home + DB`: **1899 passed /
  13 failed / 8 errors**, against base **1877 passed / 13 failed / 8 errors** --
  the same 13 failures with byte-identical names (chachanotes sync-log retention
  + v47 messages-FTS backfill; the errors are pytest tmpdir `rm_rf` teardown
  noise, same count both arms). `Tests/App + Tests/ProductionApp`: **253 passed
  / 6 failed** on BOTH arms, identical names. Collect sweep **59,680 ->
  59,702** (+22), zero collection errors on either side.
- `ruff check` clean on every touched file. `ruff format --check`:
  `Tests/DB/test_pragma_settings.py` is already unformatted at base and the one
  line added to it is format-clean (verified with `--diff`); the new test file
  is formatted.
- `./scripts/preflight.sh` **green**. The diagnostic inventory reported 4 rows,
  reviewed individually with `--statements --since` the pin's own commit before
  writing: 1 is mine (`logger.debug("Event state rollback failed: {}",
  type(rollback_error).__name__)` -- a constant plus a type NAME, no path, user
  content, secret or URL); the other 3 (`RAG_Search/simplified/*`) are
  **pre-existing stale-pin drift** -- `--since` reports *no diagnostic statement
  changed in those files* between the pin's commit and now, and `git diff`
  confirms this branch touches no RAG file. See out-of-scope findings.

### Out-of-scope findings (not fixed here)

- **`ServerParityStateRepositories.close()` -- and therefore
  `EventStateRepository.close()` and `ClientNotificationsDB.close()` -- is
  called from NOWHERE in production.** No `on_unmount` wiring exists for these
  stores. The close path is correct and tested, but nothing exercises it at
  shutdown. Wiring it is a behavioural change beyond these ACs.
- **A `:memory:` event-state store is silently unreachable from Home's
  threaded refresh.** `LocalNotificationHomeActiveWorkAdapter._active_work_seams_confirmed_file_backed`
  checks `notification_service.store` and `server_event_service.local_service.store`
  but NOT `event_state_repository`. In the app's error fallback
  (`app.py:8488`) the event-state store is `:memory:` while the notifications
  store may be file-backed, so the compute is threaded and every event-state
  call raises the same-thread `ProgrammingError`, swallowed by the adapter's
  broad `except` into "Server event feed is unavailable." Pre-existing;
  unchanged by this task.
- **Two pre-existing privacy findings in `RAG_Search/simplified/rag_service.py`,
  surfaced rather than absorbed** while reviewing the diagnostic inventory rows
  this branch had to regenerate. Both are present verbatim at `68c061984`:
  - line 1341: `logger.debug(f"[{correlation_id}] Cache hit for query: '{query[:50]}...'")`
    -- the first 50 characters of the user's **search query** into a persistent
    debug sink.
  - line 1905: `f"Rejected media_db_path from config ({db_path_raw!r}): …"` --
    a config-supplied **database path** into a persistent sink.
- **`acknowledge_event` remains a check-then-act across two transactions**
  (reads the processed cursor, compares to `expected_cursor`, then records).
  Merging it would change the method's contract; the ACs say behaviour
  unchanged.

**Files**: `tldw_chatbook/Notifications/event_state_repository.py`,
`tldw_chatbook/DB/private_sqlite.py` (owner entry corrected),
`Tests/Notifications/test_event_state_repository_connections.py` (new, 17),
`Tests/DB/test_pragma_settings.py` (+1 held-connection case),
`Tests/DB/test_private_sqlite_interop_owners.py` (+1 file-owner case),
`Docs/security/production-diagnostic-inventory.json`,
`backlog/docs/sqlite-private-owner-inventory.md`,
`backlog/docs/lessons-testing-evidence.md`.
