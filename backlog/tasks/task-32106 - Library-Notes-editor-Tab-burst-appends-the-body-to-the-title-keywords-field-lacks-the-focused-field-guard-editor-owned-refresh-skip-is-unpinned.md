---
id: TASK-32106
title: >-
  Library Notes editor: Tab burst appends the body to the title; keywords field
  lacks the focused-field guard; editor-owned refresh skip is unpinned
status: To Do
assignee: []
created_date: '2026-09-08 22:43'
labels:
  - library
  - notes
  - bug
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32062 fix rounds (PR #2531): with title + Tab + body typed in one uninterrupted burst, Tab's focus move lands after the burst and the body text is appended to the title (nothing is lost; 1 s gaps behave); `apply_session_state` writes `wide_keywords.value` with no `has_focus` guard (`library_notes_canvas.py:1687-1689`) — the same stale-snapshot clobber class as the title bug one field over, and `_NOTE_EDITOR_INPUT_IDS` excludes keywords; nothing pins that the editor-owned refresh skip is instance-scoped to the work pane (moving the guard to the screen would freeze the list silently); the list scroll-offset re-apply is skipped while the editor owns focus. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A title, Tab, body burst lands the body in the body field
- [ ] #2 The keywords field is treated as its own authority while focused, like title and body
- [ ] #3 A test pins that a sync while the editor has focus still repaints the list pane
- [ ] #4 The list scroll offset survives a sync that lands mid-edit at compact widths
<!-- AC:END -->
