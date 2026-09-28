---
id: TASK-32693
title: Add Plugins navigation and source-qualified browsing
status: To Do
assignee: []
created_date: '2026-09-16 04:32'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32689
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users an accessible plugin destination with stable browsing state and understandable partial readiness.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Plugins is reachable through shell navigation, command palette and canonical Settings without reassigning existing shortcuts or using deprecated settings surfaces.
- [ ] #2 Installed, Browse and Marketplaces views provide bounded search/details, scope/effective activation, compatibility axes and source provenance with explicit empty/error/stale states.
- [ ] #3 Search, page, selection and focus survive wide/narrow transitions; obsolete responses and row reordering cannot retarget an action.
- [ ] #4 Token-governed controls and keyboard-only 80x24 and 120x35 layouts keep scope, disabled explanations and actions reachable without automatic remote previews.
<!-- AC:END -->

## Renumbering provenance

This uncommitted planning task moved from TASK-32690 to TASK-32693 after the final allocation scan found an independent TASK-32690 claim in the workflows authoring worktree. The three-task delivery tail was moved together so all dependency IDs remain lower than the dependent task. No existing foreign task was renamed.
