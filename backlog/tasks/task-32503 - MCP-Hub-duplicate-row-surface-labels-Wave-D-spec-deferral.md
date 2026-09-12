---
id: TASK-32503
title: 'MCP Hub: duplicate-row surface labels (Wave D spec deferral)'
status: Done
assignee: []
created_date: '2026-09-11 23:10'
updated_date: '2026-09-11 23:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred label disambiguation from the Wave D spec (ADR-148): the same tool names legitimately appear under both the external-MCP inventory group (builtin:tldw_chatbook) and the Console agent group (local:__local__/virtual CLI) as two separate permission domains -- label them 'tldw_chatbook (external MCP)' and suffix the Console-side group '(Console agents)' so the pairing reads as deliberate, plus the pinned two-row test from the spec.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 builtin inventory group renders as tldw_chatbook (external MCP) in matrix sections, Tools Server column and preview,Console agent group label carries (Console agents),A pinned test renders one tool name under both groups with both surface labels visible,All pinned label tests updated in the same change
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implements the deferral recorded in Wave D's plan self-review / TASK-32455 notes. Branch `feat/mcp-hub-ux-wave-d`.
- Two constants carry the change: `Agents/local_tool_provider.LOCAL_SERVER_LABEL` -> "Local workspace, web, and Watchlists (Console agents)" and `hub_tool_catalog.builtin_tools_from_inventory`'s label -> "tldw_chatbook (external MCP)" — every surface (matrix sections, Tools Server column, preview) derives from these.
- Deliberately NOT suffixed: `VIRTUAL_CLI_SERVER_LABEL` ("Virtual CLI (read-only)" — Console-only tools, no cross-surface duplication) and the approval-path audit label "Local workspace" in unified_control_plane_service (different surface, no duplication, pinned approval copy).
- The spec's pinned two-row test landed (`test_same_tool_under_both_surfaces_renders_both_labels`: fs_read under both sections, both surface labels rendered). Pinned label tests updated in the same change: the local-catalog assertions, the distinctness pin (`tldw_chatbook (external MCP)` vs `Built-in (agent runtime)`), and the Tools-mode truncation fixture.
- ADR check: covered by ADR-148 (this is the label half of its decision 1; no new decision).
- Verification: workbench 349 passed (full); permissions/servers/rail/tools 187 passed; permission resolution/store/control-plane 187 passed; inspector 284 passed; doc-contract unchanged at the pre-existing 39 failures.
<!-- SECTION:NOTES:END -->
