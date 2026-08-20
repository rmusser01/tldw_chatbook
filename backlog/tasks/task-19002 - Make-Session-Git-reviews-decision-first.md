---
id: TASK-19002
title: Make Session Git reviews decision-first
status: To Do
assignee: []
created_date: '2026-08-20 07:40'
labels:
  - notes
  - git
  - ux
  - accessibility
dependencies:
  - TASK-19001
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Lead commit and push reviews with the facts required for authorization while retaining implementation and recovery evidence through progressive disclosure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Commit and push reviews lead with `What`, `Where`, `Impact`, and `Recovery` sections.
- [ ] #2 Every fact that can change authorization remains visible without opening technical details, including exact destination, candidate, lease, hooks, transport, and publication scope.
- [ ] #3 Technical details are collapsed by default, contain audit-only evidence, and remain keyboard-operable with correct focus restoration.
- [ ] #4 Existing Git trust, staging, commit, push, uncertainty, and cancellation contracts remain unchanged.
- [ ] #5 The decision facts and disclosure remain contained, scrollable, and keyboard-safe at 40x20.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/035-file-notes-session-git-index-controls.md`, `backlog/decisions/038-file-notes-guarded-session-commit.md`, `backlog/decisions/039-file-notes-guarded-session-push.md`
Reason: this task reorganizes existing review evidence without changing Git authorization or execution policy.
