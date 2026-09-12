---
id: TASK-32290
title: 'Docs: approvals and MCP permissions pages match shipped behaviour'
status: To Do
assignee: []
created_date: '2026-09-10 19:17'
labels:
  - docs
  - approvals
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
agent-runs-and-tools.md lists four decisions while the card has five, says Always allow is MCP-only while local workspace tools also get it, quotes a path-warning string that differs from the code, and its approval-card SVG shows the pre-TASK-1846 single-line layout. mcp.md is a stub with no explanation of Inherit, Allow, Ask, Off, Space cycling, the kill switch, or that an explicit tool-level Allow bypasses the risk floor. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both pages describe the five decisions, their scopes and where to undo them.
- [ ] #2 The path-warning text matches the code and the approval-card SVG is regenerated from the current layout.
- [ ] #3 mcp.md documents the Permissions matrix states, Space cycling, the kill switch and the risk-floor rule for explicit tool-level Allow.
- [ ] #4 Both pages carry an updated 'Verified against' stamp.
<!-- AC:END -->
