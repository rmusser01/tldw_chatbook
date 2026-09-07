---
id: TASK-292
title: Media search imports deleted UI.MediaWindow module — always errors
status: Done
assignee: ['@claude']
created_date: '2026-07-17 21:30'
labels: [bug, media, library]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-283 (PR review sweep of the debounced-search paths): Event_Handlers/media_events.py's perform_media_search_and_display imports UI.MediaWindow, a module deleted by the Library redesign — the import raises, so media search always renders its "Error loading…" state on current dev. The search itself (search_media_db) works; only the display path is broken. Fix: route the display through the current Library media surface (or the appropriate current widget), and add a regression test that exercises the real handler end-to-end (the deleted-module import would have been caught by any unmocked call).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media search displays results through a module that exists; the handler imports cleanly
- [x] #2 An unmocked test drives perform_media_search_and_display end-to-end (import failure class pinned)
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: unmocked end-to-end test drives the real handler with a real file-backed MediaDatabase; confirm it fails on dev (renders "Error loading: No module named ...").
2. Remove the dead show-deleted block (the legacy UI.MediaWindow import + show_deleted_items read); default show_deleted=False, the only surviving behavior since no current surface exposes the toggle.
3. Verify the regression test passes post-fix and media suites stay green.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Root cause: perform_media_search_and_display's body imported the legacy UI.MediaWindow (deleted in the Library redesign) to read a show_deleted_items toggle; the ModuleNotFoundError escaped the local `except QueryError` and was swallowed by the function's broad `except Exception`, rendering every search as "Error loading". `show_deleted_items` exists nowhere else in the codebase, so the block was fully dead: it is removed and show_deleted stays False (deleted items were already excluded in practice for the pre-redesign UI too, since the toggle's owner no longer mounted).

Regression net: Tests/Media_DB/test_media_search_display.py drives the REAL handler end-to-end (real file-backed MediaDatabase so the asyncio.to_thread path runs, minimal Textual harness app, no mocks) — verified to FAIL against the unfixed handler and pass after. Any future deleted-module import inside the handler resurfaces as the "Error loading" row the test asserts against.

Files: tldw_chatbook/Event_Handlers/media_events.py (block removal), Tests/Media_DB/test_media_search_display.py (new).
<!-- SECTION:NOTES:END -->
