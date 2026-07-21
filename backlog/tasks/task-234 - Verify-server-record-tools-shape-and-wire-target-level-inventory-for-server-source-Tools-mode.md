---
id: TASK-234
title: >-
  Verify server-record tools shape and wire target-level inventory for
  server-source Tools mode
status: To Do
assignee: []
created_date: '2026-07-16 15:19'
updated_date: '2026-07-21 01:44'
labels:
  - mcp
  - phase4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tools mode derives server-source tools from each external-server record's embedded 'tools' list (raw /mcp/hub pass-through) — the only prior consumer used it as a len() fallback, so the dict shape (name/description/inputSchema) is unverified against tldw_server. Bare targets are excluded entirely even though ServerUnifiedMCPService.get_inventory() exists. PHASE 4 PREREQUISITE before server-source execution (from PR #639 Task-5 review + final review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 tldw_server external-record tools dict shape confirmed and documented,Bare-target tools surface in Tools mode via target-level inventory,Empty-state diagnosis distinguishes 'no tools' from 'inventory not fetched'
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DISPOSITION (2026-07-21, program close): server-source tool execution was DEFERRED ENTIRELY by user decision during Phase 6 planning — the MCP Hub program shipped with local+builtin execution only; server-source tools are display-only (copy updated accordingly in PR #722). The shape-verification half of this task remains valid IF server execution is ever revived: MCPUnifiedClient.execute_tool still lacks a per-call access context, no recorded real-server tools fixture exists, and ServerUnifiedMCPService has no execute wrapper. Keeping open at low priority as the revival prerequisite.
<!-- SECTION:NOTES:END -->
