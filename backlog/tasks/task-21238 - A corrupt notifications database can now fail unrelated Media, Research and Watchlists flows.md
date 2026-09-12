---
id: TASK-21238
title: >-
  A corrupt notifications database can now fail unrelated Media, Research and
  Watchlists flows
status: To Do
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

- [ ] A corrupt or unopenable notifications database does not fail a media ingest job, a research run, or a watchlists inbox refresh — the flow completes and the notification is skipped
- [ ] A test injects a corrupt notifications store and asserts each of those three flows still completes
- [ ] The two Watchlists inbox workers cannot take the app down on an escaping exception
- [ ] The lazy-store race test covers all six stores, or the five uncovered stores are shown to be exercised by the one covered case
- [ ] TASK-21105's boot-file count (27) does not regress
