---
id: TASK-32213
title: >-
  Library Media: a 0-result filter hides the whole toolbar including the active
  type facet
status: Done
assignee: []
created_date: '2026-09-10 14:53'
updated_date: '2026-09-10 19:23'
labels:
  - library
  - media
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With `type: pdf` and a filter that matches nothing the canvas keeps only the title, the filter box and the miss sentence; `type: pdf`, `sort:`, `Export…`, `Trash`, `Select` and `Review these` are gone, so the type that produced the empty page cannot be seen or reset from the canvas (guide: 'A filtered empty page keeps its submitted type, query, or collection visible until you choose the reset action'). Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The toolbar (at least the type facet, sort and Trash) stays visible on a 0-result page
- [x] #2 The miss sentence names the active type when one is set, with a reset action
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: a 0-result filtered page keeps type:/sort:/Trash and names the type
2. empty_copy ladder in library_media_state.py names the active type alongside the query
3. fresh_zero block stops returning early; only the recovery button stays conditional
4. Re-run the fresh-zero pins
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The fresh-zero branch in library_media_canvas.py no longer returns before the toolbar: only its recovery BUTTON is conditional (Show all types whenever a type is set; Import media only for a genuinely empty source, so task-31224's 'a filter miss never suggests Import' holds). A second early return sits just below the load-failure callout so the LIST furniture -- row viewport, 'No media item selected.' placeholder, 'Item 0-0 of 0' pager -- stays away exactly as before; only the toolbar is new. library_media_state.py's empty_copy ladder gains the both-facets case: "No media of type 'pdf' matched “x” in titles, content or keywords."

Two knock-on changes the fall-through forced, both deliberate:
- The review-dismiss and analyze receipts can now also compose on a fresh-zero page (fresh_zero excludes delete_receipt_count but not those two). Kept: an Undo affordance surviving a filter to zero is the right behaviour, and it is the same argument task-31635 used for the Sets opener.
- 'Review these' would have painted live on a page with nothing to review (its gates consult load failure and staleness, never rendered_count) while 'Select' correctly painted '○ Select' with a reason; pressing it only raised a toast. It now takes the same rendered_count == 0 gate, with the '○' marker and LIBRARY_MEDIA_REVIEW_EMPTY_TOOLTIP. 'Export…' stays live on purpose: its scope is the type, not the query.

Keeping the toolbar also made #library-media-type-filter a permanently enabled entry-focus fallback on an empty page, which would have landed entry focus on the facet the user did NOT type into -- the symptom task-32214 was filed for. _library_media_empty_list_fallback_target now puts #library-media-filter first while a query is applied, honouring the intent its own docstring already recorded; both stale docstrings in library_screen.py were corrected.

Pins moved with the behaviour, none loosened: test_media_fresh_zero_distills_to_one_recovery_action now pins the EXACT body-action inventory (stronger than the old len == 1) and still holds the recovery budget at one; test_library_paged_empty_recovery_is_painted_and_keyboard_reachable's media leg swapped an absent selector; test_background_recompose_restores_focus_on_a_filtered_empty_media_list keeps every assertion but loses its premise -- its docstring now says the strict 'channel cannot land' leg is UNPINNED and names the rider that re-pins it.

Live: 235x52 and 100x30. Files: library_media_canvas.py, library_media_state.py, library_screen.py, test_library_crit9_media_list.py, test_library_multiselect_media.py, test_library_shell.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
