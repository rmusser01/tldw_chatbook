---
id: TASK-24404
title: Activate nightly deep workflow on default branch
status: To Do
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-29'
labels:
  - ci
  - infrastructure
  - github-actions
priority: high
dependencies:
  - task-24403
---

## Description

Activate the reviewed nightly deep test workflow from the repository's default
branch so GitHub actually schedules the full-tree matrix against `dev`, without
promoting unrelated development changes to `main`.

## Acceptance Criteria

- [ ] Default-branch `main` contains the exact reviewed `.github/workflows/nightly-deep.yml` file from `dev` and no unrelated files from TASK-24403
- [ ] The workflow owns both `schedule` and `workflow_dispatch`, explicitly checks out `dev`, and retains the five-environment full-tree matrix
- [ ] GitHub registers the workflow from `main`, and one live manual dispatch reaches a terminal truthful verdict
- [ ] A real scheduled event creates the expected matrix against `dev` and reaches terminal truthful verdicts before this task is marked Done

ADR required: no

ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`

Reason: this task is the operational default-branch activation of the schedule
architecture already decided by ADR-103; it introduces no additional boundary.

Design: `Docs/superpowers/specs/2026-08-29-fast-pr-lane-design.md`

## Definition of Done

- [ ] Every acceptance criterion is checked
- [ ] YAML parsing and exact-file comparison pass
- [ ] The activation PR contains only the dedicated workflow file
- [ ] Manual and real scheduled runs provide terminal evidence
- [ ] Review feedback is resolved and implementation notes are recorded
