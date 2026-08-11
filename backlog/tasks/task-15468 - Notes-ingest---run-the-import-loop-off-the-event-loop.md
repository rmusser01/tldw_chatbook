---
id: TASK-15468
title: Notes ingest: run the import loop off the event loop
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `Event_Handlers/note_ingest_events.py:306` ("Import Selected Notes Now") dispatches a coroutine worker whose own comment (`:638-641`) states it runs on the main event loop: per file a sync parse, per note a sync `notes_service.add_note(...)` transaction (INSERT + FTS triggers + commit/fsync), plus sync template JSON I/O — O(files x notes), serially. Importing dozens/hundreds of notes is a guaranteed multi-second full-app freeze, exactly matching the reported symptom on this surface.

Fix direction: `thread=True` worker with `call_from_thread` for the UI updates (the callbacks already marshal results), or `to_thread` the per-file/per-note body. Preserve per-note error accounting and the preview flow. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The UI processes input while a large import runs (evidence: interaction during an N-hundred-note import)
- [ ] #2 Import results, preview, template handling, and per-note failure accounting unchanged (tests)
- [ ] #3 Import wall-time not materially regressed (before/after)
<!-- AC:END -->
