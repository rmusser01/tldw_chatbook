---
id: TASK-28243
title: 'Review sets - Phase 4: set picker (resume / switch / dismiss)'
status: To Do
assignee: []
created_date: '2026-09-02 22:29'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28240
  - TASK-28241
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A lightweight picker to resume, switch between, or dismiss saved review sets (design: backlog/docs/design-library-review-sets.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A picker opened from the media list lists saved sets with name + progress (X of M, reviewed N); selecting one activates it and loads at its cursor
- [ ] #2 The picker can dismiss (soft-delete) a set and reopen a completed one; activating a set deactivates the previously active one (one-active invariant)
- [ ] #3 Reuses the Library choice-strip / picker idioms; no new rail row required for v1
<!-- AC:END -->
