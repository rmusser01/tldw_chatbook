---
id: TASK-1338
title: 'Local agent tools phase 1: plumbing + fs_list pilot'
status: To Do
assignee: []
created_date: '2026-08-05 00:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md. Plan: Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md. ADR: backlog/decisions/032. Build LocalToolProvider + approval-hook generalization + workspace-root config, proven end-to-end with fs_list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 LocalToolProvider lists/schemas/invokes fs_list through the agent runtime loop
- [ ] #2 Approval card gates fs_list with allow/session/always/deny wired to the permission store under local:__local__
- [ ] #3 Kill switch and fail-closed no-callback paths return the pinned refusal strings
- [ ] #4 workspace_root and local_tools_enabled config keys coerce and default correctly
- [ ] #5 All new tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md
<!-- SECTION:PLAN:END -->
