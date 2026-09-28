---
id: TASK-32694
title: Review and manage plugin operations across UI surfaces
status: To Do
assignee: []
created_date: '2026-09-16 04:33'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32693
  - TASK-32675
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete install/configure/update/remove workflows with review-bound actions and honest recovery status.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review binds exact immutable inputs and offers Install disabled or explicit workspace enablement; configuration and explicit execution/testing remain distinct.
- [ ] #2 Updates, rollback, scoped disable, uninstall and data deletion expose affected workspaces, active blockers, stopping, persistence failure, cleanup-pending and recovery outcomes.
- [ ] #3 Close/Back preserves in-session draft and focus without cancelling or approving; explicit cancellation follows the current durable phase and cannot retarget after navigation.
- [ ] #4 Library Skills and MCP handoffs preserve installation/component/workspace identity with ownership labels and guarded edits; mounted/live isolated-profile flows verify all lifecycle actions.
<!-- AC:END -->

## Renumbering provenance

This uncommitted planning task moved from TASK-32691 to TASK-32694 after the final allocation scan found an independent TASK-32690 claim in the workflows authoring worktree. The three-task delivery tail was moved together so all dependency IDs remain lower than the dependent task. No existing foreign task was renamed.
