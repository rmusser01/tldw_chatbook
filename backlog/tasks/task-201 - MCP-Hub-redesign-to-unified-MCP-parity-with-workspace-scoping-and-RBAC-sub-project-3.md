---
id: TASK-201
title: >-
  MCP Hub redesign to unified-MCP parity with workspace scoping and RBAC
  (sub-project 3)
status: To Do
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
- [ ] #1 Own design spec approved before implementation,Governance/permissions surfaces expose the existing capability+rules+approvals primitives,Tool availability is configurable per workspace and globally,Server-first defaults contract respected per task-88
<!-- AC:END -->
