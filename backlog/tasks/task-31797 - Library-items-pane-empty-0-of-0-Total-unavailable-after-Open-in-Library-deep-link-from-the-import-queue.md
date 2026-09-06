---
id: TASK-31797
title: >-
  Library items pane empty ('0 of 0 - Total unavailable') after 'Open in
  Library' deep-link from the import queue
status: Done
assignee: []
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 15:24'
labels:
  - bug
  - ui
  - library
  - ingest
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Import a local .md via the Library import flow, wait for the job's done state, click 'Open in Library': the reader opens the item correctly but the middle Items pane shows '0 of 0 - type: None' / 'No page loaded - Total unavailable / Page boundary is unknown' and never recovers on its own. Clicking 'Media (1)' in the left rail populates it. A user arriving via the deep-link sees an apparently empty library beside their open item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 'Open in Library' deep-link lands with the items list populated (at minimum the opened item's page loaded).
- [x] #2 Regression test for the deep-link path asserting a non-empty items page.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: import a .md, click Open in Library, confirm middle Items pane stays '0 of 0 · type: None / No page loaded'.\n2. Map the deep-link (_open_library_item_by_id media branch) vs the Media rail path; find the missing browse-page load.\n3. Add the rail's _request_library_media_browse + _request_library_media_facets to the media branch (focus_identity=None to keep viewer focus).\n4. Add a regression test asserting the deep-link issues exactly one media browse request.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the media branch of LibraryScreen._open_library_item_by_id (the ingest 'Open in Library' deep-link route via _navigate_to_media, plus the sibling Search/RAG evidence + landing-hub 'Open' routes) jumped straight to the media viewer but never asked the browse controller to load a page. Unlike the Media rail-row path (_select_library_rail_row_after_source_admission), which calls _request_library_media_browse + _request_library_media_facets, the deep-link left LibraryMediaBrowseController.freshness = 'uninitialized', so the middle Items pane rendered '0 of 0 · type: None / No page loaded · Total unavailable'.

Fix: mirror the rail's browse+facets request in the media branch (after the reader detail worker, before _apply_library_media_active_surface). focus_identity=None keeps focus on the just-opened viewer rather than yanking it to the first list row.

Reproduced live (tmux, fresh scratch profile) before the fix and confirmed fixed after: Open in Library now lands with the Items pane showing '1-2 of 2', the opened item loaded, type facets populated.

Test: Tests/UI/test_library_open_in_library_items_t31797.py drives the non-entry_origin media branch via an unbound call and asserts exactly one _request_library_media_browse (focus_identity=None) + one facets request; mutation-verified (fails without the fix).

Files: tldw_chatbook/UI/Screens/library_screen.py; Tests/UI/test_library_open_in_library_items_t31797.py; Docs/User_Guide/library/import-and-export.md (verified-against stamp).
<!-- SECTION:NOTES:END -->
