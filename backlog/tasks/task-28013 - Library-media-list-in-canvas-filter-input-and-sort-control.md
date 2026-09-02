---
id: TASK-28013
title: Library media list - in-canvas filter input and sort control
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
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
- [ ] #1 Typing in an in-canvas filter narrows the media list in place
- [ ] #2 Sort order is user-selectable from the canvas
- [ ] #3 Active filter and sort state is visible and easy to clear
<!-- AC:END -->
