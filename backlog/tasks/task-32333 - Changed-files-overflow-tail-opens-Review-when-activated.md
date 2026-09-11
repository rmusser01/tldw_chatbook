---
id: TASK-32333
title: >-
  Changed-files overflow tail opens Review when activated
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
UX review C4. The '+N more - open Review' tail and the pruned-history line are plain Statics phrased as instructions (console_changed_files_section.py ~196-211) -- false affordance. Make the tail an activatable control that opens the Review screen; reword the pruned line as passive status.

Filed from the 2026-09-10 Console rail UX review (review item C4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The overflow tail is keyboard-focusable and clickable, and activating it opens the Review screen
- [ ] #2 The pruned-history line reads as status, not an instruction
- [ ] #3 Row-click behavior and the 12-row cap are unchanged
<!-- AC:END -->
