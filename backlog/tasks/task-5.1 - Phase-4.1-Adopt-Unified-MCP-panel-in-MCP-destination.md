---
id: TASK-5.1
title: Phase 4.1 Adopt Unified MCP panel in MCP destination
status: Done
assignee: []
created_date: '2026-05-04 03:19'
updated_date: '2026-05-04 03:25'
labels:
  - unified-shell
  - phase-4
  - mcp
  - service-adoption
dependencies: []
parent_task_id: TASK-5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the top-level MCP destination use the existing Unified MCP management panel so users can inspect and operate local or server MCP context from the primary shell instead of seeing a disabled placeholder.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP route renders Unified MCP source server scope section and action controls
- [x] #2 Legacy tools_settings alias still resolves to the MCP destination
- [x] #3 The old disabled MCP management placeholder is removed from the top-level MCP route
- [x] #4 Focused automated tests cover the adopted MCP destination behavior
- [x] #5 QA evidence documents functional behavior visual usability residual risks and verification output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing destination shell tests proving the MCP route mounts UnifiedMCPPanel controls and no longer exposes the disabled placeholder.
2. Replace the placeholder MCPScreen content with UnifiedMCPPanel while preserving BaseAppScreen route identity.
3. Preserve view-state save restore and runtime backend refresh delegation for the embedded panel.
4. Add Phase 4 QA evidence and roadmap links.
5. Run focused MCP destination and unified panel regression tests plus git diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Embedded the existing `UnifiedMCPPanel` directly in the top-level MCP destination and preserved `tools_settings` as an MCP alias. Added MCP screen view-state save and restore, runtime-backend refresh delegation, a tooltip for the exposed Unified MCP action button, focused destination regression coverage, and Phase 4 QA/roadmap tracking.
<!-- SECTION:NOTES:END -->
