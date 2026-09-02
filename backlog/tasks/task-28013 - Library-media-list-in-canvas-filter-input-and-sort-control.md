---
id: TASK-28013
title: Library media list - in-canvas filter input and sort control
status: Done
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 13:53'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Narrowed after live re-verification 2026-09-02: the in-canvas filter SHIPPED on dev (a "Filter media" input narrows the list in place; pager updates; Ctrl+U clears). Remaining scope is the SORT control: MediaBrowseScope supports seven sorts (library_media_state.py:72) but the canvas exposes none of them. Side observation from the run, worth a look while here: narrowing the filter also switches which item is loaded in the Reader (filtering to talk2 loaded talk2; clearing restored talk1) - confirm that is intended.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing in an in-canvas filter narrows the media list in place
- [x] #2 Sort order is user-selectable from the canvas
- [x] #3 Active filter and sort state is visible and easy to clear
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adds a browse SORT chooser to the media list (the in-canvas filter half shipped earlier on dev). Mirrors the proven Prompts/Notes/type-filter choice-strip pattern exactly (task-14902 requires the strip, not a cycle): a Sort button opens a direct-pick strip (compose_library_choice_strip) offering Newest/Oldest/Title A-Z/Title Z-A (MEDIA_SORT_CHOICES; relevance is query-only so excluded, date_* dedups last_modified_* locally). Picking a new order re-fetches page one via _request_library_media_sort (dataclasses.replace(applied, sort_by=..., page=1)); the active sort persists in the scope. Wired through the shared seams: _library_open_choice_strip (footer 'enter: choose sort'/'esc: cancel' + Escape close), mutual-exclusivity with the type chooser, _begin_library_media_mutation clear, and the state builder (new LibraryMediaCanvasState.sort_by/sort_choices_visible defaults). GOTCHA the test caught: _close_open_library_choice_strip has a hard-coded visibility_attr->canvas_kind dict that needed the new sort attr or Escape KeyError'd. Tests: sort-applies + mutual-exclusivity/escape integration + 28025 5-button-fit extension. Files: Library/library_media_state.py, Widgets/Library/library_media_canvas.py, UI/Screens/library_screen.py, Tests/UI/test_library_shell.py + test_library_media_side_by_side.py.
<!-- SECTION:NOTES:END -->
