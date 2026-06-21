---
id: TASK-127
title: Console workspace resume and saved conversation UAT
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
Verify and harden Console's workspace rail for saved conversation listing, workspace switching, and resuming prior chats so users can return to active work without leaving Console or losing workspace context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saved Console conversations appear in the active workspace conversation list with enough metadata to identify which chat will be resumed.
- [ ] #2 User can resume a saved conversation from the workspace rail and see the corresponding transcript and active conversation state.
- [ ] #3 Switching workspaces updates the visible saved conversation list without leaking workspace-specific conversations into other workspace rails.
- [ ] #4 The Default workspace remains usable for normal chat while preserving the local-only and file-tools-disabled policy.
- [ ] #5 Existing workspace-scoped new-chat creation from TASK-126 remains available and does not regress while resume behavior is added.
- [ ] #6 Leaving and returning to Console preserves or restores the selected workspace and resumed conversation state.
- [ ] #7 Unavailable server sync and handoff paths remain explicitly labeled as WIP or unavailable.
- [ ] #8 Mounted regression coverage verifies saved conversation listing, resume, Default workspace behavior, and cross-workspace visibility boundaries.
- [ ] #9 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Parallel Ownership

Owns workspace/conversation state, Default workspace policy, workspace conversation list behavior, and resume from saved conversations. Avoid rewriting transcript rendering, provider settings, or message-action internals except through existing Console seams.

ADR required: yes, only if this task changes workspace persistence, ownership, or handoff contracts.
ADR path: backlog/decisions/ or N/A after implementation planning.
Reason: Workspace ownership and persistence rules are long-lived cross-module contracts; UI-only wiring against existing contracts does not require a new ADR.
