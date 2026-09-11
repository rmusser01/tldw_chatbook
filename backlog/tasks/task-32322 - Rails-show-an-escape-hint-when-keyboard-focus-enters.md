---
id: TASK-32322
title: >-
  Rails show an escape hint when keyboard focus enters
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review A3. Tab is region-locked per CONSOLE_TAB_REGIONS (chat_screen.py ~656-661); F6/Esc are the exits but nothing teaches that at the point of need. When focus enters a rail, surface a brief contextual hint naming the exit keys, without permanently spending a row.

Filed from the 2026-09-10 Console rail UX review (review item A3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When keyboard focus moves into either rail, a transient or dismissible hint identifies F6 (next pane) and Esc (composer) as exits
- [ ] #2 The hint does not appear for mouse-only interaction and does not change rail layout height persistently
- [ ] #3 Hint copy matches the actual bindings (truthfulness rule, TASK-2154 FR-06)
<!-- AC:END -->
