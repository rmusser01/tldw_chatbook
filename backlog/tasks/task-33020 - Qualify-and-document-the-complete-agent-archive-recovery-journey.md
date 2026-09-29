---
id: TASK-33020
title: Qualify and document the complete agent archive recovery journey
status: To Do
assignee: []
created_date: '2026-09-27 15:26'
labels: []
dependencies:
  - TASK-33019
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the integrated recovery journey through production paths and document its limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Targeted integrated tests prove lifecycle, permission, chronology, concurrency and cancellation requirements.
- [ ] #2 A disposable-profile live app run demonstrates both entry points and original-chat opening without implicit sends.
- [ ] #3 User guides, ADR links and scoped lint and UI governance evidence describe delivered behavior accurately.
<!-- AC:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
