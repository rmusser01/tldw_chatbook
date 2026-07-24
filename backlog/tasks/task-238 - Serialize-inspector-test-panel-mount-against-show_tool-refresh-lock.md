---
id: TASK-238
title: Serialize inspector test-panel mount against show_tool refresh lock
status: To Do
assignee: []
created_date: '2026-07-16 15:19'
labels:
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCPInspector._mount_test_tool_panel/_close_test_tool_panel serialize against each other via a worker group but not against show_tool's _refresh_lock. Probed 9x without misbehavior (single await point), but no shared invariant enforces it — defense-in-depth before Phase 4 adds more panel states (PR #639 Task-6 review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Panel mount/close and show_tool share one serialization mechanism,Back-to-back selection + panel-open sequences covered by a regression test
<!-- AC:END -->
