---
id: TASK-38200
title: >-
  Console: a rail search pass pulls focus back to the Conversations filter after
  the user moved it
status: To Do
assignee: []
created_date: '2026-10-30 06:04'
labels:
  - console
  - left-rail
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact.** Every debounced pass of the Conversations 'Filter visible titles' search ends by scheduling a refocus of the filter input, whether or not focus is still there. A user who types a filter and then clicks a row, the composer or another section before the pass lands has focus pulled back into the filter, so the next keys go to the search instead of where they were aimed.

**Evidence.** Code reading during the TASK-33622.6 review: the workspace controller's flat-search pass calls call_after_refresh(_focus_console_workspace_conversation_search) on each of its exits (UI/Console_Modules/workspace.py, the empty-query, local-rows, error and settled branches), and ChatScreen._focus_console_workspace_conversation_search focuses the input unconditionally. Behaviour predates TASK-33622.6; since that task the input keeps its focus across rebuilds, so the refocus is no longer needed to restore it while the user is typing. Not yet reproduced live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When focus has moved out of the Conversations filter (to the composer, a row, or another section) before a debounced search pass finishes, the pass leaves focus where the user put it
- [ ] #2 While the user is still typing in the filter, and after Clear, focus stays in the filter (no regression of TASK-33622.6)
- [ ] #3 A Pilot test on the real ChatScreen types into the filter, moves focus to the composer before the debounce fires, and asserts the composer still has focus after the pass; it fails on the pre-fix code
<!-- AC:END -->
