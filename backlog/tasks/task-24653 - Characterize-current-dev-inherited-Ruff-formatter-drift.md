---
id: TASK-24653
title: Characterize current-dev inherited Ruff formatter drift
status: To Do
assignee:
  - '@codex'
created_date: '2026-08-30 15:39'
labels:
  - maintenance
  - formatting
  - quality
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-30-task-24653-ruff-formatter-debt-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-22514 proved that its closeout introduced no Ruff formatter regressions while leaving a historical 61-file residue on its pinned base. Re-census current origin/dev and define conflict-safe atomic cleanup batches so formatter debt stops obscuring feature-owned changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pinned current `origin/dev` census records every Python file failing the repository-supported Ruff format check.
- [ ] #2 The current census is compared with the 61-file TASK-22514 historical residue and every difference is explained.
- [ ] #3 A mechanically checked batch manifest assigns every current failure exactly once to an atomic cleanup record whose own acceptance criteria require behavior preservation and an eventual repository-wide zero-exit Ruff format check.
- [ ] #4 TASK-24653 changes no Python source and its documentation diff plus the Backlog task-ID guard pass.
<!-- AC:END -->
