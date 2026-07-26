---
id: TASK-719
title: Settings workspace rows label stutter circular recovery copy and stale destination references
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - settings
  - copy
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Overview rows read like debug output ('Workspace default: Workspace: Workspace 3 (workspace-local-3); ...', 'Sync safety: Collections: Sync: dry-run only; ...'). The recovery copy says to open 'the matching Settings category or destination' without ever naming which destination owns workspace management. Code fallback copy still points at 'Library > Workspaces', a mode that no longer exists (it is now a disclosure under the Library rail's Details section). Findings m1/m2; captures cap-11, cap-27.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace and sync rows read as label: value without repeated label fragments
- [x] #2 Settings copy names the concrete surfaces that own workspace switching and management
- [x] #3 No user-facing copy references the retired Library > Workspaces mode
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. De-stutter _workspace_default_label and _sync_safety_label values; rewrite the Overview recovery copy to name owning surfaces; replace stale "Library > Workspaces" references.
2. Update pinned assertions in test_settings_configuration_hub.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Workspace default row value no longer repeats "Workspace:/Authority:/Sync:" prefixes ("Research (research); authority server-backed; sync ready"); sync-safety join strips each surface's leading "Sync:" ("Collections: dry-run only"). Overview recovery copy now names the owners: switch in Console (Alt+W), manage in Library > Details > Workspace, sync from the owning surfaces. Stale "Library > Workspaces" copy (a retired mode) replaced in chat_screen switch-toast and display_state S3 recovery with the real affordances (rail New button / Library > Details > Workspace). Hub suite: 248 green (theme-editor mount test is a suite-load timing flake, passes standalone); Workspaces suite 106 green.
<!-- SECTION:NOTES:END -->
