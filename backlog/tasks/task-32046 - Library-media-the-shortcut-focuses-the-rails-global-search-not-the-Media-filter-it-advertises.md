---
id: TASK-32046
title: >-
  Library media: the / shortcut focuses the rail's global search, not the Media
  filter it advertises
status: Done
assignee: []
created_date: '2026-09-08 14:36'
updated_date: '2026-09-08 15:09'
labels:
  - library
  - media
  - ux
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P1 (lead). On the Media list the footer advertises '/ focus search', but pressing / focuses the rail's global 'Search Library…' field two panes away, not the Media 'Title/keyword…' filter the user is looking at, so a filter query does nothing to the list until the misdirection is noticed. Keyboard-first is the product's core and this misfires the headline gesture; it is worst for keyboard-only users, whose focus silently leaves the pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing / while the Media list (or another canvas with its own filter) is active focuses THAT canvas's filter input, not the rail's global search
- [x] #2 The footer hint's target matches where focus actually lands
- [x] #3 A pin drives / on the Media list and asserts the focused widget is the Media filter input, at 235x52 and 100x30
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`/` on the Media (and Prompts) list now focuses that canvas's own filter (`#library-media-filter` / `#library-prompts-filter`) instead of falling through to the rail's global `#library-search-input`. Added an early per-canvas filter route in `LibraryScreen.on_key`'s slash branch (before the screen-wide rail grab, in a try/except that falls through when the filter is absent), mirroring the existing notes/conversations routes; other rows and the rail search are unchanged. Pin at 235x52 AND 100x30 (red first: landed on the rail search). Files: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_shell.py.
<!-- SECTION:NOTES:END -->
