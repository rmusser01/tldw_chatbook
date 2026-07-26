---
id: TASK-700
title: Open a server-ingested item from the ingest queue
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 13:58'
updated_date: '2026-07-26 22:00'
labels:
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A server ingest finishes with its content in the server's library, not this machine's, so the queue row's 'Open in Library' stays withheld. The server does report the id of the row it created, so the item is addressable -- there is just no affordance that opens it in the server-scoped Library view. Users who import on the server currently have no route from a finished job to the thing it produced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A finished server-origin job offers a way to view the item it created
- [x] #2 The action opens the item in a server-scoped view rather than looking for a local row
- [x] #3 A job whose server result carries no usable id offers no such action
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A finished server ingest now offers 'View on server', and the item it created is reachable.

The server reports the id of the row it made; that id was being discarded. It is now captured as remote_media_id -- deliberately not media_id, which means a row in this machine's DB. The two id spaces are unrelated, so a server id stored there would point 'Open in Library' at a wrong or absent local row.

Schema v4 adds the column by plain ALTER TABLE rather than a rebuild: nothing changes an existing CHECK constraint.

The row gate requires origin=='server' AND a present id, so a job the server reported no id for offers nothing rather than dead bait. 'Open in Library' stays withheld for server jobs; the two actions are never both offered. The handler flips the viewer's detail fetch to server mode, and every local route into the viewer clears that flag so one server view cannot make the next local item resolve remotely.

A collision this nearly shipped with: the button first reused the class .library-ingest-open and an id starting with 'library-ingest-open-'. The existing handler selects on that class and strips that prefix to recover a job id, so it would have caught this button and parsed the bogus id 'server-ingest-job-3' -- opening nothing, from the wrong handler. Both are now distinct, guarded by a test that inspects the id constructions rather than raw source.

VERIFIED LIVE, end to end against a running server: a real submit produced a real completed job whose result carried media_id 1125; the real reconciler folded that into a real registry, leaving remote_media_id='1125' with the local media_id still None; the real state builder then produced can_open=False, can_open_on_server=True. Fetching that id through the server-mode path returns MediaDetailResponse(media_id=1125) -- the exact row the ingest created, so the action resolves what it claims to.

Separately verified through real persistence: two finished server jobs written to a v4 database, read back through the hydration path, gave True and False respectively.

Both clauses of the gate are mutation-checked.

Files: DB/Library_Ingest_Jobs_DB.py (v4), Library/library_ingest_jobs.py, Library/server_ingest_reconcile.py, Library/library_ingest_state.py, Widgets/Library/library_ingest_canvas.py, UI/Screens/library_screen.py.
<!-- SECTION:NOTES:END -->
