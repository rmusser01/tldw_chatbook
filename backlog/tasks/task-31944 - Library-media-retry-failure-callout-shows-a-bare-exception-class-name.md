---
id: TASK-31944
title: Library media - retry failure callout shows a bare exception class name
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR E final review M-4 (2026-09-05): the media browse controller's _retry_failure_reason falls back to the exception's class name when the exception carries no message, so the user reads 'Couldn't retry - RuntimeError'. A short map for the classes that actually occur (timeout, connection, database) would give a reason a user can act on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A retry failure caused by a timeout, a connection error or a database error shows a human-readable reason instead of the exception class name
- [ ] #2 An unmapped exception still produces a callout with a usable fallback reason
- [ ] #3 The mapping and the fallback are pinned
<!-- AC:END -->
