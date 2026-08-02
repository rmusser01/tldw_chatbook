---
id: TASK-2013
title: >-
  Library ingest silently swallows duplicate files
status: To Do
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description (the why)

A byte-identical file at a different path takes
`add_media_with_keywords`'s duplicate-skip path, which returns
`(None, None, "Media '<title>' already exists. Overwrite not enabled.")`.
The writer (`app.py` `_run_library_ingest_queue`) only falls back to
`get_media_by_url`, which misses because the URLs differ — so the job is
marked done with no `media_id`: a "✓ done" queue row that created nothing,
says nothing about the duplicate, and (because `can_open` requires a
`media_id`) has no "Open in Library" action. Live evidence: a folder ingest
reported 5 done while the Media count reached only 4, with no explanation
anywhere. Found in the 2026-08-02 ingest UAT (critique snapshot
2026-08-02T21-04-04Z).

## Acceptance Criteria (the what)

- [ ] A duplicate ingest resolves the existing media item's id (content-hash
      fallback after the URL miss), so its done row carries "Open in
      Library" and opens that item.
- [ ] The duplicate row's progress line states the file was already in the
      Library instead of impersonating a fresh ingest.
- [ ] A genuinely fresh ingest keeps its current "Ingested <path>" message.
