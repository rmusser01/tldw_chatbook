---
id: TASK-31575
title: Library Trash - an empty permanent-delete reconcile still flips the page stale
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Permanent-delete reconcile is a content no-op that still marks the page stale when has_authority is false, so the Trash pane re-fetches for nothing. The controller-side fix (empty reconcile is a no-op) was barred by wave 4 PR C's no-controller-refactor constraint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An empty reconcile leaves the page fresh
- [ ] #2 A controller test pins it
<!-- AC:END -->
