---
id: TASK-32353
title: >-
  Library Export: the default 'quality: thumbnail' silently ships previews, and
  the bundle is never listed before writing
status: Done
assignee: []
created_date: '2026-09-11 06:16'
updated_date: '2026-09-11 07:49'
labels:
  - library
  - export
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The export canvas defaults to 'quality: thumbnail — keeps a small preview image instead of the full file', rendered at the same weight as 'sort'; nothing lists the items or estimates the size before Export bundle (A cap 50). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export defaults to full fidelity
- [x] #2 The canvas shows what the bundle will contain (item count, fidelity, estimated size) before the button is pressed
- [x] #3 Pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Flip DEFAULT_MEDIA_QUALITY to 'original'; update the two default pins.
2. Extend the export counts worker with a media title + byte preview query (parameterised, same connection, never raises).
3. Add consequence_line + contents_lines to LibraryExportFormState and its builder; reuse _count_phrase and lift format_last_export_line's KB rounding into one shared formatter.
4. Yield both lines in library_export_canvas.compose, display-toggled, above the submit button.
5. Docs + live verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DEFAULT_MEDIA_QUALITY is now "original": the form opens at full fidelity, so a bundle only loses content when asked. The two pins asserting the old default were updated (Tests/Library/test_library_export_state.py, Tests/UI/test_library_choice_strips.py, whose direct-pick half now targets "thumbnail" so it still exercises a real change). Chatbooks' own defaults (chatbook_models.py:546, chatbook_creator.py:268) were checked and deliberately left alone -- different surface, own callers, and ChatbookCreationWizard.py:944 passes an explicit value.

AC#2: preview_export_scope() (library_export_scope.py) reads the in-scope media items' titles and their stored content's UTF-8 byte total -- one parameterised query on the same connection, run on the counts worker beside count_export_scope, and it never raises out of that worker (a missing seam, a missing table or a too-large selection all degrade to the empty preview). It lands through _apply_library_export_counts under the same staleness guards as the counts, into a new LibraryExportState.preview field that resets with them. build_library_export_form_state gained consequence_line + contents_lines; the canvas yields both display-toggled above the submit button, and the counts-landing patcher owns them (recompose discipline).

Deviation from the plan's pinned string, with evidence: live on the seeded profile a selection estimated at "about 9 KB" wrote a 3,606-byte archive whose receipt read "4 KB". The estimate now says "about N KB before compression" -- it counts content going in, the receipt stats the zip coming out, and without the qualifier the two read as a contradiction.

Formatters: format_export_bytes() lifts format_last_export_line's own KB rounding into one function both call, so the estimate and the receipt can never round differently; the count phrase reuses _count_phrase from library_export_scope.

Files: tldw_chatbook/Library/library_export_scope.py, library_export_state.py; tldw_chatbook/UI/Library_Modules/library_export_state.py, library_export_controller.py; tldw_chatbook/UI/Screens/library_screen.py (counts worker + apply + builder); tldw_chatbook/Widgets/Library/library_export_canvas.py; Tests/UI/test_library_crit10_export.py (new), Tests/Library/test_library_export_scope.py, test_library_export_state.py, test_library_export_execution.py, Tests/UI/test_library_choice_strips.py, test_library_export_receipt.py; Docs/User_Guide/library/import-and-export.md.

Known limitation (honest, not silent): an "everything" scope spans four sources and only media is sizeable up front, so it renders "size known once it runs" and no contents list. The media-only scopes -- selection or whole-source, with or without a type filter -- get both.
<!-- SECTION:NOTES:END -->
