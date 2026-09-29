---
id: TASK-33017
title: Coordinate confirmed archive restoration without navigation
status: To Do
assignee: []
created_date: '2026-09-27 15:26'
labels: []
dependencies:
  - TASK-33014
  - TASK-33016
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the selected resource scope with truthful outcomes while preserving current work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both entry points use exact-target one-shot confirmation and the approved lifecycle action matrix.
- [ ] #2 Stale state, name conflicts, duplicates and cross-store partial failures report actual resource outcomes.
- [ ] #3 Cancellation, session closure and shutdown retain started-write ownership; restoration never activates or sends.
<!-- AC:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
