---
id: TASK-34887
title: Reconcile TASK-21251 Windows startup fix verification and closure
status: To Do
assignee: []
created_date: '2026-09-27 15:12'
labels:
  - actor-packs
  - windows
  - maintenance
dependencies: []
references:
  - >-
    backlog/tasks/task-21251 -
    Keep-Windows-startup-alive-when-Actor-Pack-cleanup-is-unsupported.md
  - backlog/tasks/task-21252 - Support-secure-Actor-Pack-import-on-Windows.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the stale In Progress record for the implemented Windows Actor Pack startup guard using current targeted verification and the repository full-suite policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shipped startup and preflight behavior is verified against the existing TASK-21251 acceptance criteria, with dated targeted test and static-check evidence.
- [ ] #2 The interrupted repository-wide run is recorded accurately and is reconciled with the current targeted-test policy; any remaining defect is documented rather than inferred.
- [ ] #3 TASK-21251 status and notes match the verified outcome while preserving prior implementation evidence and keeping native Windows import under TASK-21252.
<!-- AC:END -->
