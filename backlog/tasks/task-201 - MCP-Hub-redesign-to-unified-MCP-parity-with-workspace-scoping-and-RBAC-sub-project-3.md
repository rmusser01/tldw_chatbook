---
id: TASK-201
title: >-
  MCP Hub redesign to unified-MCP parity with workspace scoping and RBAC
  (sub-project 3)
status: Done
assignee: []
created_date: '2026-07-12 13:16'
labels:
  - ux
  - mcp
  - workspaces
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Third spec of the user-confirmed decomposition. Redesign the MCP screen beyond the current thin panel (overview/inventory/external_servers) to full unified-MCP functionality: manage MCP servers/tools/settings/profiles, per-workspace vs global tool availability (WorkspaceOperation.TOOL_USE eligibility + the <prefix>:<workspace>:<scope> config-key precedent), permissions/RBAC surfaces over the existing governance primitives (capability registry, local rules, approval queue, server permission profiles). Must respect the task-88 constraint: Chatbook does not own persisted MCP defaults until the tldw_server unified-MCP contract is confirmed. Requires its own brainstorm/spec before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Own design spec approved before implementation,Governance/permissions surfaces expose the existing capability+rules+approvals primitives,Tool availability is configurable per workspace and globally,Server-first defaults contract respected per task-88
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Landed as the MCP Hub program's sub-project 3: commit 1d154d9c0a 'feat(agents): MCPToolProvider bridging the agent thread to main-loop MCP execution (task-201)' added Agents/mcp_tool_provider.py (427 lines) + Tests/Agents/test_mcp_tool_provider.py (663 lines); the governance/permissions surfaces it required shipped as Phases 4-6 (mcp_permissions_mode.py, permission_store.py, approval queue, audit mode).
