---
id: TASK-2013
title: >-
  Library ingest silently swallows duplicate files
status: In Progress
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
- [x] The duplicate row's progress line states the file was already in the
      Library instead of impersonating a fresh ingest.
- [x] A genuinely fresh ingest keeps its current "Ingested <path>" message.

## Implementation Notes

The parse payload never carries `content_hash` — `add_media_with_keywords`
computes `sha256(content)` internally — so the writer's fallback (and the
`mark_done(content_hash=...)` stamp) had always been fed `None`. The writer
now mirrors the DB's exact hash computation from `payload["content"]`, falls
back from the URL lookup to `get_media_by_hash`, and labels the duplicate's
progress "Already in Library — matched an existing item; nothing new was
imported." Side effect: done jobs now carry a real `content_hash` stamp.
Files: `tldw_chatbook/app.py` (`_run_library_ingest_queue`),
`Tests/Library/test_library_ingest_runner.py`
(`test_duplicate_content_at_different_path_resolves_existing_media_id`).
Verified: new test red→green (red failed on exactly `None == 1`); full
runner file 67/67.
