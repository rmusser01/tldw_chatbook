---
id: TASK-31584
title: >-
  Library review-set chip - remove the dead progress helper and fix the at_last
  wording
status: To Do
assignee: []
created_date: '2026-09-05 03:24'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_active_review_set_progress is production-dead (tests only) and the at_last footer chip says finish review when an earlier item in the set is still unreviewed (wave 4 PR B Task 1 deferred minors).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dead helper is removed or used
- [ ] #2 The at_last chip wording is honest when an earlier item is unreviewed
- [ ] #3 A test pins the wording
<!-- AC:END -->
