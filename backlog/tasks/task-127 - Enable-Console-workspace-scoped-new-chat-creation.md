---
id: TASK-127
title: Enable Console workspace-scoped new chat creation
status: To Do
assignee: []
created_date: '2026-06-21 00:35'
labels:
  - console
  - workspaces
  - uat
dependencies:
  - TASK-128
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console's workspace rail support creating and resuming chat conversations in the active workspace so users can continue work inside the current operating context without leaving Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can create a new Console chat from the active workspace context rail when a non-default workspace is active.
- [ ] #2 Created Console sessions are associated with the active workspace and appear in that workspace conversation list.
- [ ] #3 Switching workspaces updates the visible conversation list without leaking workspace-specific conversations into other workspace rails.
- [ ] #4 The Default workspace remains usable for normal chat while preserving the local-only and file-tools-disabled policy.
- [ ] #5 Saved conversations can be resumed from the workspace conversation list after leaving and returning to Console.
- [ ] #6 Unavailable server sync and handoff paths remain explicitly labeled as WIP or unavailable.
- [ ] #7 Mounted regression coverage verifies active-workspace creation, Default workspace creation, resume, and cross-workspace visibility boundaries.
- [ ] #8 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Parallel Ownership

Owns workspace/conversation state, Default workspace policy, workspace conversation list behavior, and resume from saved conversations. Avoid rewriting transcript rendering, provider settings, or message-action internals except through existing Console seams.

ADR required: yes, only if this task changes workspace persistence, ownership, or handoff contracts.
ADR path: backlog/decisions/ or N/A after implementation planning.
Reason: Workspace ownership and persistence rules are long-lived cross-module contracts; UI-only wiring against existing contracts does not require a new ADR.
