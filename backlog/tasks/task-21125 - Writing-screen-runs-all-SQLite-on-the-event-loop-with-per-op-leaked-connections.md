---
id: TASK-21125
title: >-
  Writing screen runs all SQLite on the event loop with per-op leaked
  connections
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-24 00:27'
labels:
  - performance
  - database
  - writing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21125).

`Writing_Interop/local_writing_service.py:56-78`: ~45 `with self._connect()` sites open a fresh
connection per operation (sqlite3's `with conn:` is a transaction manager, not a closer -
GC-only leak), and the entire call chain from `UI/Writing_Window.py` / writing_controller has
zero thread offload - tree clicks and autosave run open + query + commit on the Textual event
loop, each open paying the private-seam's ~4 artifact verifications.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The service holds a thread-local connection (WAL+NORMAL already set) and closes it explicitly on shutdown
- [x] #2 Controller calls route through asyncio.to_thread; a thread-assert (or log probe) confirms no SQLite on the loop from this screen
- [x] #3 Writing screen behavior unchanged - existing tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline the writing test set on base fb0a9601e (tee to test-logs/).
2. Red-first probes in Tests/Writing_Interop/test_local_writing_service.py and Tests/UI/test_writing_screen.py: connection-open counter over N ops, pragma read-back on the LIVE held connection (journal_mode=wal, synchronous=1, isolation_level None), per-thread connection affinity, close()-then-reuse re-arm, close() waits for an in-flight operation on another thread (shutdown race), rollback failure must not mask the original error, and a loop-thread SQLite assertion driven through WritingController.
3. LocalWritingService: held per-thread connection (dict keyed by thread ident under a lock so close() can reach every thread), opened via connect_private_sqlite with check_same_thread=False and isolation_level=None; explicit _transaction() context manager doing BEGIN / COMMIT / ROLLBACK replacing all 44 'with self._connect() as conn:' sites; _ensure_schema stays once-per-service (TASK-21105) and _init_schema closes its own raw connection instead of leaking it.
4. Lifecycle gate: operations register under a condition variable; close() sets a closing flag, waits for in-flight operations to settle, closes every held connection, then re-arms so a later operation transparently reopens. Rollback failures are suppressed with a type-name-only debug log so the original exception survives.
5. Thread offload: WritingController gains a _call(method, *args) seam that awaits coroutine functions directly and routes plain callables through asyncio.to_thread; WritingScopeService._service_for_mode returns a thread-offloading proxy for non-coroutine backend methods so the local (sync) service never runs on the loop while local_service identity is preserved.
6. Shutdown: app on_unmount peeks the local_writing_service slot (never constructs) and closes it, next to the Library ingest jobs store close.
7. Error surface: confirm a corrupt/unopenable writing DB still degrades to a status message rather than crashing compose/mount.
8. Evidence: pragma read-back, open-count before/after, thread assertion, timing figure. Run the writing test files plus app lifecycle, full --collect-only sweep, ruff, ./scripts/preflight.sh.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`LocalWritingService` now holds ONE connection per thread for the life of the
service instead of opening (and GC-leaking) a fresh one per operation, and the
whole local backend runs on a worker thread instead of the Textual loop.

**Connection lifecycle.** `_connect()` returns a held connection from a
`dict[thread_ident, Connection]` (not a `threading.local`: shutdown has to reach
connections it does not own), opened through the same `connect_private_sqlite`
seam with `check_same_thread=False` and `isolation_level=None`. All 43
`with self._connect() as conn:` sites became `with self._transaction() as conn:`,
an explicit BEGIN / COMMIT / ROLLBACK context manager -- sqlite3's connection
context manager is a *transaction* manager, not a closer, which is what leaked
the connections in the first place. `_init_schema` now closes its own raw
connection too. The schema pass stays once per SERVICE (`_schema_ready`, from
TASK-21105) -- stricter than once-per-connection, so no per-transaction census
was reintroduced. `_transaction()` is re-entrant (a nested call joins the open
transaction rather than issuing a second BEGIN), and `:memory:` -- which shares
one connection across threads -- serialises transactions under an RLock.

**Shutdown race (the TASK-21101 review class).** Operations register in a
condition-variable gate; `close()` sets a closing flag, waits for operations
owned by OTHER threads to settle (bounded at 5 s, then warns rather than
hanging), closes every held connection, and re-arms so a later operation
transparently reopens. Regression test
`test_local_writing_service_close_waits_for_an_in_flight_operation` parks a
transaction mid-flight on a worker thread and asserts `close()` blocks for it;
mutation-verified -- replacing the settle wait with `settled = True` reds it with
exactly the 21101 signature, `ProgrammingError('Cannot operate on a closed
database.')`. A failing ROLLBACK is swallowed with a type-name-only debug log so
it can never mask the original exception
(`test_local_writing_service_transaction_error_is_not_masked_by_rollback`, also
mutation-verified).

**Thread offload.** The blocking frame was NOT the controller: every
`WritingScopeService` method is `async def`, so a controller-level `to_thread`
would have moved zero work in the shipped app. `_service_for_mode` now returns a
`_ThreadOffloadedBackend` proxy whose `__getattr__` dispatches non-coroutine
callables through `asyncio.to_thread` (async backends pass straight through, so
the server backend pays no hop) -- one edit covering ~70 `_maybe_await` call
sites, with `scope.local_service` identity preserved for the packaging wiring
assertion. `WritingController` additionally gained a `_call(method, *args)` seam
replacing `_maybe_await(service.method(...))`, which offloads a synchronous
backend wired directly to the controller.

**Shutdown wiring / error surface.** `TldwCli._close_local_writing_service()`
peeks the `local_writing_service` slot (never constructs one to close it --
TASK-21103 pattern) and is called from `on_unmount` beside the Library ingest
jobs store close. Per the TASK-21105 caution, an unopenable writing DB still
surfaces at first *use*, not at compose/mount: a corrupt file yields
`status_message == 'file is not a database'` and `load_projects() == []`, pinned
by `test_unopenable_writing_database_degrades_to_a_status_message`.

**Evidence** (logs in `test-logs/task-21125-*`):

- Baseline (base `fb0a9601e`, writing set): 79 passed / 0 failed.
- Red-first: 8 of the 11 new tests failed on base; the other 3 are guards.
- Probe, 20 iterations of 2 reads + 1 write through the real
  controller-shaped graph: **180 connection opens, all on `MainThread`,
  0.1289 s -> 0 opens, every statement on `asyncio_0`, 0.0057 s (~23x)**.
  Total opens including warm-up: 3 (schema + one per thread).
- Pragma read-back on the LIVE held connection: `journal_mode=wal`,
  `synchronous=1`, `isolation_level=None`; WAL confirmed persistent in the file
  via an independent connection.
- After: writing set 92 passed; combined writing + private-sqlite +
  service-composition run 141 passed / 0 failed; `Tests/App` +
  import-closure guards 187 passed. `Tests/ProductionApp` has 4 failures,
  A/B-proven identical on base `fb0a9601e` (transformers cache, reactive
  maturity scanners, retired-destination state -- none writing-related).
  `Tests/Packaging/test_installed_distribution.py` errors are environmental
  (this venv has no `setuptools`, so every sdist/wheel build fixture errors).
- Full `--collect-only` sweep: 57,694 collected, 4 pre-existing optional-dep
  collection errors (`torch`, `playwright`).
- `ruff check` / `ruff format --check` clean on every file touched (`app.py`
  is not ruff-format-clean at base, so it was hand-matched and verified to
  produce no formatter diff in the added region).
- `./scripts/preflight.sh` green after reviewing the 5 diagnostic-inventory
  rows (4 new calls in `local_writing_service.py`, 1 in `app.py`) -- all log
  constants or `type(exc).__name__` only, never the DB path or user content.

**Files**: `tldw_chatbook/Writing_Interop/local_writing_service.py`,
`tldw_chatbook/Writing_Interop/writing_scope_service.py`,
`tldw_chatbook/UI/Writing_Modules/writing_controller.py`,
`tldw_chatbook/app.py`, `Tests/Writing_Interop/test_local_writing_service.py`
(+7), `Tests/Writing_Interop/test_writing_scope_service.py` (+2),
`Tests/UI/test_writing_screen.py` (+4),
`Tests/DB/test_private_sqlite_interop_owners.py` (the writing owner factory now
returns a real closer), `Docs/security/production-diagnostic-inventory.json`,
`backlog/docs/lessons-testing-evidence.md`.

**Deviation from the plan / not done.** The plan said the offload would land in
the controller; the controller seam alone is provably insufficient (the scope
service is async), so the offload landed at `WritingScopeService`'s dispatch
point as well. Also surfaced but deliberately NOT touched (behavioural change,
outside these ACs): `WritingController` calls six methods --
`get_project_structure`, `autosave_scene`, `search_project`, `assign_chapter`,
`move_scene`, `restore_version_to_working_state` -- that exist on neither
`WritingScopeService` nor `LocalWritingService`, so those screen paths only work
against the test fake.
<!-- SECTION:NOTES:END -->
