---
id: TASK-1977
title: 'Change review: auto-register nested repos as tracked sub-roots (fast-follow)'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
dependencies:
  - TASK-1976
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the nested-repo hole: repos detected inside a root become their own tracked sub-roots — own shadow repos keyed by their canonical paths, excluded from the parent's shadow repo, bounded depth. The review aggregates parent + sub-root diffs per turn; the TASK-1976 banner disappears for auto-registered children.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An edit inside a nested repo now appears in the turn's review, attributed to the sub-root
- [ ] #2 The parent's shadow repo excludes the child (no gitlink churn rows)
- [ ] #3 Sub-root discovery is bounded (depth/count) with disclosure when the bound truncates
- [ ] #4 Removing the child repo un-registers its sub-root via existing GC
<!-- AC:END -->
