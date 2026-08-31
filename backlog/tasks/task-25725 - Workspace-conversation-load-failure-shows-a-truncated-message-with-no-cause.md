---
id: TASK-25725
title: Workspace conversation load failure shows a truncated message with no cause
status: To Do
assignee: []
created_date: '2026-08-31 05:09'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the workspace conversation list fails to load, the rail shows a clipped string and a bare Retry control. Two distinct root causes are collapsed into one generic sentence, the real exception is swallowed into a debug log, and the visible text is cut off by the rail width so the user cannot read even the generic message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail error is legible at the rail's actual width
- [ ] #2 Distinct failure causes produce distinct user-facing messages
- [ ] #3 The underlying exception is recorded at a level the user can be pointed to
<!-- AC:END -->
