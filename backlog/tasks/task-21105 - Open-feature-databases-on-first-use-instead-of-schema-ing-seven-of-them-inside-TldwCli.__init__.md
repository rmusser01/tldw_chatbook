---
id: TASK-21105
title: >-
  Open feature databases on first use instead of schema-ing seven of them inside
  TldwCli.__init__
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 10:53'
labels:
  - performance
  - startup
  - database
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21105).

Seven feature databases are created and schema'd synchronously inside `TldwCli.__init__` for
features a user may never touch: research (5 tables + migrations, app.py:6948), notifications
(app.py:7068), event_state (10 DDL) + sync_state (16 DDL) server-parity stores (app.py:7081),
writing (16 DDL, app.py:6723), kanban (24 DDL, app.py:7277 - zero UI consumers found at all),
notes_sync_state (start path). Each is file create + WAL setup + executescript + fsync traffic,
serial, pre-paint. The lazy seam already exists: `BaseDB.__init__(initialize_schema=False)`
(DB/base_db.py:43).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the listed stores opens (and creates its schema) on first feature use, not during app construction; feature behavior on first use is unchanged
- [x] #2 Per-store regression tests assert no DB file exists after a boot that never touches the feature
- [x] #3 Boot construction time before/after recorded in the task (isolated-profile probe)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: isolated-profile boot probe (TldwCli() time + files-created census) on base 8e949873e\n2. Consumer census per store (writing, research, kanban, notifications, event_state/sync_state) via parallel searches; notes-sync leg untouched (TASK-21112)\n3. Shape: in-class deferred schema (BaseDB initialize_schema=False seam / move _init_schema out of __init__ into a single-flight _ensure_schema hooked at the connection chokepoint) -- NOT app-level lazy properties, because (a) sync_scope_service and the notification services are read directly by later boot wiring, (b) Tests/ProductionApp AST-pins _wire_* calls in __init__, (c) the app_factory harness patches get_*_db_path only during construction so path resolution must stay eager\n4. Convert 6 classes: LocalWritingService, LocalResearchService, LocalKanbanService, ClientNotificationsDB, EventStateRepository, SyncStateRepository\n5. Red-first per-store tests: construction creates no DB file; first op creates schema and works; close-before-use safe. Plus one subprocess isolated-profile boot test asserting all six files absent after TldwCli()\n6. Run per-store suites + Tests/App + RuntimePolicy + navigation/composition tests; full --collect-only sweep; A/B any red against base\n7. Re-run boot probe; record before/after in task notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Six store classes now defer file creation + schema DDL from `__init__` to a
single-flight `_ensure_schema()` hooked at each class's one connection
chokepoint. Construction still resolves the path eagerly (test harnesses
patch `get_*_db_path` only around app construction) and `:memory:`
instances keep the old eager init (no disk cost; their single cached
connection must stay bound to the constructing thread -- the app's parity
`:memory:` fallback would otherwise bind to whichever worker thread touched
it first). **Zero app.py changes**: every wiring line, identity assertion,
AST composition pin, and `getattr` boot-path read is untouched, which is
why in-class deferral was chosen over the 21103 app-level-property shape.
The notes_sync_state leg was left alone entirely (TASK-21112).

Converted (chokepoint in parens):
- `Writing_Interop/local_writing_service.py` `LocalWritingService` (`_connect`)
- `Research_Interop/local_research_service.py` `LocalResearchService` (`_connect`; migrations now also run at first use)
- `Kanban_Interop/local_kanban_service.py` `LocalKanbanService` (`connect`)
- `Notifications/client_notifications_db.py` `ClientNotificationsDB` (`_get_connection`, via `initialize_schema=False`)
- `Notifications/event_state_repository.py` `EventStateRepository` (same)
- `Sync_Interop/sync_state_repository.py` `SyncStateRepository` (same)

Each conversion splits the old connection factory into `_open_connection()`
(raw) + the ensure-wrapped public method, so `_initialize_schema` can run
inside the lock without deadlocking; a failed ensure leaves the flag unset
so the next operation retries.

**Consumer census** (full detail in the review agents' reports; verified
against base `8e949873e`): writing -- one reader
(`UI/Writing_Window.py:30`, getattr-None-tolerant, Writing screen compose);
research -- Research window + `/research` console command, all
getattr-guarded; the one boot-path read (`app.py:7772` sync_scope patch
loop) is getattr-with-default over an eagerly-present attribute, unchanged;
notifications -- first store touch is Home active-work compose or a
dispatched notification/reminder; the dispatch service holders all inject
at boot but call only on events; event_state -- only
`NotificationsScopeService` server-gated paths (`wrong_source` policy
denial fires before the repo in local mode); sync_state -- no boot reader;
first any-mode touches are Settings mount (`list_write_sync_promotion_
states`) and Library Collections decoration, both off-thread and both now
paying the one-time schema at that moment instead of every boot.

**Kanban retire-candidate evidence (confirmed)**: exhaustive census found
ZERO consumers of `local_kanban_service`/`kanban_scope_service`/
`server_kanban_service` outside `app.py` wiring (521-524, 7453-7471) and
tests -- no UI screen, route, tab, widget, event handler, MCP tool, or
agent surface reads them. The only runtime effect of the eager
construction was its own boot cost (24 DDL + FTS5 probe into a DB nothing
reads). Lazy implemented here; retirement is a separate task.

**Boot evidence** (isolated scratch profile, subprocess `TldwCli()`
construction, no mount):
- Files created at boot: 35 -> 27. Gone: `tldw_chatbook_research.db`,
  `tldw_chatbook_writing.db`, `tldw_chatbook_kanban.db`,
  `tldw_chatbook_notifications.db` (+`-wal`/`-shm`),
  `tldw_chatbook_event_state.db`, `tldw_chatbook_sync_state.db` -- 8 files,
  ~90 DDL statements, 6x WAL setup + fsync traffic off every boot.
- Construction time: cold first run 2.238 s (base) vs 0.580 s (converted;
  cache-warm confound). Honest warm-vs-warm alternating A/B (3 runs each):
  base 0.453/0.460/0.471 s vs branch 0.436/0.445/0.451 s -- ~17 ms (~4%)
  on a warm APFS SSD; the durable win is the removed fsync/file traffic
  (cold boots, first runs, slow disks) and profile dirs no longer
  accumulating never-used stores.

**Tests**: new `Tests/DB/test_feature_store_lazy_open.py` (18 tests: per
store, construction-creates-no-file / first-use-works /
close-before-use-safe, + a 4-thread single-flight race) and
`Tests/App/test_boot_no_feature_db_files.py` (subprocess scratch-profile
boot census, asserts all six filenames + sidecars absent while ChaChaNotes
proves the boot ran). Red-first verified by running both files against
base: 12 failed there (every construction pin + the boot census), 7 passed
(the first-use happy paths -- feature behavior unchanged). Four
construction-time pins updated to first-use triggers with their subjects
intact: `test_private_sqlite_interop_owners.py` writing/research factories
(seam + PrivatePathError contracts now measured at first use),
`test_held_connections.py` notifications reuse test (warm-up before the
open-counter; subject stays reuse-across-inserts), the four research
migration tests, and the sync_state column-migration validation test.

**Semantics shift (deliberate, documented; corrected in review)**: a
corrupt/unreadable store now surfaces at first feature use instead of at
boot. For the stores that had app-level fallbacks this IS a new error
surface, not just a timing move: on base, construction failure fell back
to `:memory:` (notifications, event_state/sync_state) or `None`
(writing/research), so corrupt-DB errors could NEVER reach those
operation paths -- the fallback (or the None-guard in the consumer)
absorbed them at boot. On this branch the first operation on a corrupt
store raises to its caller. The `/research` worker was the one consumer
this actually broke (app teardown -- fixed in the review round below);
hardening the notifications dispatch helpers + watchlists inbox workers
for the same class is filed separately as the MAJOR-2 follow-up. Kanban
previously had NO catch (a corrupt kanban DB crashed `__init__`); it can
no longer crash boot and still has zero consumers.

Test evidence: per-store suites Writing_Interop/Kanban/Research/
Research_Interop 298 passed; Notifications/Sync_Interop/RuntimePolicy/
active-work/watchlists-shell 847 passed; Subscriptions+Study_Interop 936
passed; DB suites (seam/pragma/inventory/held-conns) 123 passed; wiring
pins (screen-navigation wiring tests, test_app_research_wiring,
audio-handoff teardown) 126 passed; Tests/App+Tests/ProductionApp 241
passed with the same 4 pre-existing reds as base (A/B'd:
test_llm_destination_actions, test_reactive_ownership_maturity x2,
test_retired_destination_root_state); phase39 library-collections file is
13F+3E identically on base. Full `--collect-only`: 56545 collected, same 5
pre-existing collection errors as base (TTS chatterbox, library file
notes, 3x Confluence).

**Review fix round** (verdict: mechanism passed, FIX-FIRST on one
regression + two smalls):
- MAJOR-1 (confirmed regression): `/research` with a corrupt research DB
  exited the app. The handler's `local_service is None` guard
  (chat_screen.py) is dead code under lazy open -- construction succeeds
  now -- and `launch_run` (the store's first use, where the corrupt file
  raises) sat OUTSIDE the worker's try, in a worker with Textual's
  default `exit_on_error=True`. Fix: moved `launch_run` inside the
  worker's guard (keeps the logged-warning degrade; the None-guard stays
  for skeletal/test apps). Regression test
  `test_research_worker_survives_a_corrupt_research_store`
  (Tests/UI/test_console_research_command.py) drives the real handler
  with a real `LocalResearchService` over a genuinely corrupt file;
  mutation-verified: reverting the guard makes it fail on the escaping
  `sqlite3.DatabaseError`.
- MINOR-2: `_initialize_schema` in event_state_repository.py and
  sync_state_repository.py leaked its raw file connection (`with conn`
  commits but never closes); both now close in `finally` like
  client_notifications_db.py. sync_state keeps the inner `with conn`
  transaction block -- `_record_schema_version` runs bare DML whose
  implicit transaction it commits; probe-verified the version stamp (4)
  survives close+reopen.
- Honesty: the "no new error class reaches callers" claim corrected
  above (wrong for the three stores whose app-level fallbacks used to
  absorb corrupt-DB errors at boot). MAJOR-2 (notifications dispatch +
  watchlists inbox hardening) filed separately, not implemented here.
<!-- SECTION:NOTES:END -->
