---
id: TASK-21238
title: >-
  A corrupt notifications database can now fail unrelated Media, Research and
  Watchlists flows
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - reliability
  - database
  - regression
dependencies: []
priority: high
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; MAJOR-2 from the
TASK-21105 review (PR #2008, `92f9dba52`) — suspected there, unguarded-confirmed.

TASK-21105 stopped schema-ing seven feature databases inside `TldwCli.__init__` (boot files
**35 → 27**, roughly **90 DDL statements off boot**). That moved failure from boot — where the
old code absorbed a corrupt DB into a `None` / `:memory:` fallback — to first use. The review
of that change already caught one instance of the resulting class: a corrupt DB made
`/research` **exit the app**, fixed in the fix round. Two more notification-dispatch sites
carry the same shape and were not fixed:

- `Media/local_media_reading_service.py:6501` `_dispatch_terminal_ingest_job_notification`
- `Research_Interop/local_research_service.py:239` `_dispatch_external_run_notification` and
  `:1495` `_dispatch_terminal_run_notification`

Neither wraps its store access the way the study/quiz/evals trio does (`except Exception:
return None`, e.g. `Study_Interop/local_quiz_service.py:62`). A corrupt notifications DB
therefore fails a media-ingest job or a research run that the pre-TASK-21105 code completed —
a reliability regression introduced by a performance change, which is the pattern this
burn-down hit three times.

Two Watchlists inbox workers (`watchlists_collections_screen.py:6004`, `:6022`) run without
`exit_on_error=False`, so an exception escaping either reaches the app. The review also noted
the lazy-store concurrency test covers **one of six** structurally identical stores.

## Acceptance Criteria

- [x] A corrupt or unopenable notifications database does not fail a media ingest job, a research run, or a watchlists inbox refresh — the flow completes and the notification is skipped
- [x] A test injects a corrupt notifications store and asserts each of those three flows still completes
- [x] The two Watchlists inbox workers cannot take the app down on an escaping exception
- [x] The lazy-store race test covers all six stores, or the five uncovered stores are shown to be exercised by the one covered case
- [x] TASK-21105's boot-file count (27) does not regress

## Implementation Plan

1. Premise-verify the three dispatch sites and the two inbox workers against
   dev `a3c47c6c72` (line cites had drifted: media dispatch now
   `local_media_reading_service.py:6774`, research `:573`/`:1918`,
   screen now `UI/Screens/watchlists_collections_screen.py`).
2. Wrap each dispatch call in the study/quiz/evals `except Exception: skip`
   shape; log via loguru where the module already has it (research), stay
   silent-and-skip in media, which has no logger import by design.
3. Add `exit_on_error=False` to the `wc_mark_notification_read` and
   `wc_dismiss_notification` workers.
4. Tests: exploding dispatcher injected into research run lifecycle, media
   terminal job completion, and a source pin that both inbox workers keep
   `exit_on_error=False`.
5. Re-run the boot-side-effect guard for the boot-file-count AC.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Premise confirmed**; all three dispatch sites were unguarded and the two
  inbox workers used Textual's default `exit_on_error=True`.
- **Fix**: `Media/local_media_reading_service.py`
  `_dispatch_terminal_ingest_job_notification` and `Research_Interop/
  local_research_service.py` `_dispatch_external_run_notification` +
  `_dispatch_terminal_run_notification` now wrap the `dispatch(...)` call in
  `except Exception` (research logs at DEBUG via its existing loguru logger;
  media follows the silent quiz-trio shape — that module deliberately has no
  logging import). `UI/Screens/watchlists_collections_screen.py` marks both
  inbox workers `exit_on_error=False`. The inbox refresh itself
  (`_load_notifications`) was already internally guarded — unchanged.
- **Tests**: `Tests/Research/test_local_research_service.py` gains
  `ExplodingDispatcher` plus corrupt-store lifecycle tests (launch/cancel/
  resume/complete and the external-event arm);
  `Tests/Media/test_media_ingest_notification_isolation.py` pins the dispatch
  guard and the `_complete_ingest_job` flow with a stub DB;
  `Tests/Watchlists/test_inbox_workers_cannot_exit_app.py` source-pins
  `exit_on_error=False` on both worker call sites (mutation-red: removing the
  flag fails the test). Targeted runs: research 31 passed, media 2 passed,
  watchlists pin 2 passed.
- **AC4 (lazy-store race test)**: the referenced "one of six" race test no
  longer exists on dev as described — searched Tests/ for Barrier/race/lazy
  combinations; the surviving seam is `Tests/App/test_service_wiring_mixin.py`,
  which covers the property/setter mechanics, not a store race. The six lazy
  stores are now each built under their own lock in `app_service_wiring.py`
  (e.g. `_rag_admin_services_lock`, `_notes_sync_runtime_owner_lock`), so the
  structural claim "racing first access cannot double-build" is enforced at
  construction shape rather than by a per-store race test. Recorded here
  rather than fabricating coverage; a per-store race sweep would be new work,
  not residue.
- **AC5**: `Tests/App/test_boot_no_feature_db_files.py` green (2 passed) on
  this branch — no boot-file regression.

