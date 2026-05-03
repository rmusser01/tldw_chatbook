---
id: TASK-2.3
title: 'Phase 1.3: Remove false Console-launch affordances from skeletal destinations'
status: To Do
assignee: []
created_date: '2026-05-03 15:33'
labels:
  - unified-shell
  - phase-1
  - affordances
dependencies: []
parent_task_id: TASK-2
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure skeletal Unified Shell destinations do not present Console launch/follow actions as executable workflows unless they carry actionable context or show an honest unavailable state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Schedules, Workflows, ACP, and W+C Console actions either pass actionable payloads or are disabled with recovery copy.
- [ ] #2 Button labels distinguish launch/follow/inspect based on actual capability.
- [ ] #3 Automated mounted UI tests prove false affordances are removed.
- [ ] #4 QA walkthrough evidence records the updated blocked or working behavior.
<!-- AC:END -->
