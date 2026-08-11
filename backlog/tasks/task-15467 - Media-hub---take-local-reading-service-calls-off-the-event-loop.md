---
id: TASK-15467
title: Media hub: take local reading-service calls off the event loop
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - media
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/MediaWindow_v2.py:2387` (`_run_media_search`) and `:1188-1191` (`_load_media_item_detail`) use `run_worker(coroutine)` — which does NOT leave the event loop — around plain synchronous `db.search_media_db(...)` (FTS + row enrichment) and `db.get_media_by_id(...)`; an item click additionally runs reading-progress, document-versions, and highlights queries, 3-4 sequential sync queries on the loop per click (`Media/media_reading_scope_service.py:628-687` `_maybe_await` -> sync `Media/local_media_reading_service.py:174-255`). The page-correction path doubles the search query. Every media search, keyword filter, pagination, subview change, undelete, and list-item click blocks input.

Fix direction: thread the local mode in the scope service, mirroring `ChatConversationScopeService`'s task-283 threading — including that task's lesson: use a POSITIVE-confirmation predicate ("confirmed file-backed -> thread"), not a negative one, so unrecognized service shapes/test doubles do not silently run inline. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No synchronous DB work runs on the event loop for media search, browse, pagination, or item click (evidence)
- [ ] #2 Results, pagination, undelete, and detail loads identical (existing tests green)
- [ ] #3 Item-click latency before/after on a large media DB recorded
<!-- AC:END -->
