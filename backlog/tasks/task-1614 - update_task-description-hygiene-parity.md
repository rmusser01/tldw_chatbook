---
id: TASK-1614
title: >-
  update_task description hygiene parity
status: In Progress
assignee: []
created_date: '2026-07-31 15:10'
labels:
  - evals
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-1482 (Task 3) fixed the name-hygiene asymmetry between create_task and update_task via a shared helper, but the DESCRIPTION parameter still differs: create_task filters control characters, update_task does not. Same parity fix, same shared-helper approach.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] update_task applies the same description cleaning as create_task, via a shared helper
- [ ] A test pins a control-character description round-trip on both paths
<!-- AC:END -->
