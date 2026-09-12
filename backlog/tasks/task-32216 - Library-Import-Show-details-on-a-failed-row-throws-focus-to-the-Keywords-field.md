---
id: TASK-32216
title: >-
  Library Import: 'Show details' on a failed row throws focus to the Keywords
  field
status: Done
assignee: []
created_date: '2026-09-10 14:53'
updated_date: '2026-09-10 17:28'
labels:
  - library
  - import
  - keyboard
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing Show details expands the errno line in place, but focus lands on the `Keywords (optional)` input near the top of the form, so a keyboard user who wanted Retry next is typing into a metadata field for the next import; once per row on a folder import. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Show details toggles, focus returns to the pressed control (the discipline the reader's More strip already applies)
- [x] #2 Pinned by a test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing UI test: Show details on a failed row throws focus to #library-ingest-keywords.
2. Controller: set_focus(event.button) synchronously before the region update, and re-resolve+set_focus by id in call_after_refresh.
3. Green; run the ingest suites and compare failing-name sets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The queue-panel repaint (`_update_library_ingest_dynamic_regions` -> `queue.refresh(recompose=True)`) prunes the pressed Button; Textual then re-picks focus from what survives -- the metadata form 25 rows up. New `LibraryIngestController._refocus_library_ingest_control(control_id, update)` wraps any queue-row toggle: focus the control synchronously with `Screen.set_focus` (never `Widget.focus()`, which defers -- the `test_library_ingest_clear_focus.py` discipline), park focus at None for the rebuild and restore the same id on the QUEUE PANEL's own post-recompose hook, with `call_after_refresh` only as the no-rebuild fallback. `LibraryIngestQueuePanel` now mixes in the existing `PostRecomposeCallback` for that hook; the parking is what the pilot could not see and the live app could -- `Button.press()` settles the pump on its way out, so a screen-pump follow-up passed in tests while typing 100ms after a real click put the characters in 'Keywords (optional)' (measured, then measured clean after the fix). Wired for 'Show details' and the new grouped-row expand toggle. Files: UI/Library_Modules/library_ingest_controller.py, Widgets/Library/library_ingest_canvas.py, UI/Screens/library_screen.py (3 delegators), Tests/UI/test_library_crit9_import.py, Docs/User_Guide/library/import-and-export.md.
<!-- SECTION:NOTES:END -->
