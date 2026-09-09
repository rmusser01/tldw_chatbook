---
id: TASK-32069
title: >-
  Library rail: stale search query persists across canvases; six Study rows for
  three destinations; 20 simultaneous choices
status: Done
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 20:03'
labels:
  - library
  - rail
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail search box keeps the last query when switching canvases with no clear affordance; the Study section spends six rows ('see what carries over' under each) on three destinations; the full rail presents 15 destinations plus 5 utility controls at once. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rail search box offers a clear affordance or resets when the canvas changes
- [x] #2 The Study section uses three rows, with the carry-over hint in the staging canvas
- [ ] #3 A recorded decision on rail density (which rows show when a source is empty)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: no clear affordance, the query follows the reader across canvases, six Study rows.
2. Add an 'x' button beside the rail search Input.
3. Seed the box from the RAG query only on the Search/RAG canvas.
4. Drop the handoff rows' second meta line and state the promise on the staging canvas.
5. Record the rail-density decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: the rail box was always seeded from _rag_search_state.query, so a query typed once followed the reader onto every other canvas -- a filter that filtered nothing, with no affordance to clear it. It now seeds from _library_rail_search_value(), which returns the live query only on the Search/RAG canvas the query actually drives; the query itself is untouched, so returning restores both box and results. An 'x' compact button sits beside the box (the input takes 1fr in its new row -- at width 100% the button rendered as two blank cells, caught live and fixed in a follow-up commit). Verified live: caps/13 (x clears), caps/14 (no carry-over).

AC#2: the three handoff rows each carried a second 'see what carries over' line at height 2 -- six rail rows for three destinations, the same sentence printed three times in the primary nav. The rows are one cell each and the promise moved into LIBRARY_STUDY_HANDOFF_OWNERSHIP_COPY, which the staging canvas paints ('This page shows what carries over; generation and review run in Study.').

AC#3 is left unticked: 'a recorded decision on rail density (which rows show when a source is empty)' is a product decision about hiding navigation from the user, not a defect I can settle inside this task's scope. The two density wins above are shipped; the empty-source rule needs an owner's call and should be its own task.

Files: tldw_chatbook/Widgets/Library/library_rail.py, tldw_chatbook/UI/Library_Modules/screen_constants.py, tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundles), Tests/UI/{test_library_crit8_polish_shell,test_library_shell}.py, Tests/Widgets/Library/test_library_rail.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
