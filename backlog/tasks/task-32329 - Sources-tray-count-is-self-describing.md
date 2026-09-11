---
id: TASK-32329
title: >-
  Sources tray count is self-describing
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B5. The staged-context tray header shows a bare digit (str(state.source_count), console_staged_context.py ~58-67) with no unit; zero renders as '0' beside the word Sources. Render a self-describing count with a real empty word.

Filed from the 2026-09-10 Console rail UX review (review item B5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tray header count renders as e.g. '3 sources' / 'no sources' (word form agreeing with the count)
- [ ] #2 Existing sync_state fingerprinting still updates the count in place
- [ ] #3 Widget tests assert the new wording for 0, 1, and N sources
<!-- AC:END -->
