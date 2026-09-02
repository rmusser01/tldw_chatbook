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
MediaBrowseScope supports seven sorts and a query (library_media_state.py:41-51) but the canvas exposes only the type filter and pager; the rail search box routes to the Search/RAG canvas instead of filtering the list (library_screen.py:31234-31263). Finding one item among dozens means paging or a Search-canvas detour. Add an in-canvas filter input (the conversations canvas filter is the template) and expose sort via the existing choice-strip idiom (task-14902 type chooser).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing in an in-canvas filter narrows the media list in place
- [ ] #2 Sort order is user-selectable from the canvas
- [ ] #3 Active filter and sort state is visible and easy to clear
<!-- AC:END -->
