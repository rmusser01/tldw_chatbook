---
id: TASK-32337
title: >-
  Sources empty state offers an in-rail next action
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
UX review D1. The empty state 'No sources attached. Stage sources from Library.' (console_staged_context.py ~107-111) is guidance without an action, while the sibling scope row offers 'Narrow...' in place. Add an activatable 'Open Library...' action and resolve the docstring drift (sync_state docstring still describes an Attach button compose no longer mounts).

Filed from the 2026-09-10 Console rail UX review (review item D1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Sources tray empty state includes a keyboard-focusable action that opens the Library staging surface
- [ ] #2 The stale docstring is corrected to describe the actual compose
- [ ] #3 Non-empty tray rendering is unchanged
<!-- AC:END -->
