---
id: TASK-714
title: Workspaces cannot be renamed archived or deleted and names are forced Workspace N
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - workspaces
  - feature
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both create paths force sequential names (Workspace 1, 2, ...) and no surface offers rename, archive, or delete. Combined with silent creation (task-713) and an invisible create button (task-712), real usage accumulates indistinguishable and accidental workspaces with no way to clean up. The retired create-local-copy control id survives as a dim WIP static. Finding M3.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A workspace can be given a user-chosen name at or shortly after creation
- [x] #2 An existing workspace can be renamed from at least one surface
- [x] #3 A non-default workspace can be archived or deleted with appropriate guardrails for its conversations
- [x] #4 The Default workspace is protected from destructive lifecycle actions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red service tests: rename (blank/unknown/Default guards), archive (listing hides, active falls back to Default, Default/unknown guards).
2. Registry service rename_workspace/archive_workspace; switcher modal grows per-row Rename/Archive with typed (action, id) results; rename modal + ConfirmationDialog wiring in ChatScreen.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Service: LocalWorkspaceRegistryService.rename_workspace (blank-name and Default-protected; WorkspaceNotFound for unknown/archived) and archive_workspace (Default-protected; sets archived=1/active=0; when the archived workspace was active, ensure_default_workspace + set_active_workspace(Default) so Console always has a real context). Conversations/memberships untouched - archiving only hides the workspace from listings (list_workspaces already excluded archived). UI: ConsoleWorkspaceSwitcherModal rows are now Horizontals with compact Rename/Archive buttons (Default gets neither - rail copy and runtime rules reference it by name); dismissal is a typed ("switch"|"rename"|"archive", workspace_id) tuple; new ConsoleWorkspaceRenameModal (AUTO_FOCUS input, Enter submits); archive goes through the existing ConfirmationDialog (its confirm callback is awaited - coroutine required) with copy stating conversations stay saved in Library. Naming at creation is satisfied via rename-immediately-after (create toast + Alt+W -> Rename); a name-prompt-on-create was deliberately skipped to keep the one-press create flow. Option-row widths use 1fr + auto so lifecycle buttons cannot reproduce the TASK-712 clip class. Tests: 6 service tests + Tests/UI/test_console_workspace_lifecycle.py (3 mounted flows); 115 green across lifecycle+Workspaces suites.
<!-- SECTION:NOTES:END -->
