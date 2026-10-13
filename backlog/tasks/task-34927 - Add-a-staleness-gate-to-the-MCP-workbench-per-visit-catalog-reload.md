---
id: TASK-34927
title: Add-a-staleness-gate-to-the-MCP-workbench-per-visit-catalog-reload
status: To Do
assignee: []
created_date: '2026-10-13 14:51'
labels:
  - performance
  - ui-perf
  - wave-3-screen-switch
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP screen is non-reusable, so every visit re-mounts and re-runs the full catalog reload: service.load_context() + _collect_snapshots() + _sync_children() over every configured server (mcp_workbench.py:963-991 -> :1127-1162), with no staleness check against the previous visit's data. Switching to MCP repeatedly pays the full reload each time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-entering MCP with unchanged config and services skips the full catalog reload (staleness check),Explicit refresh path still forces a full reload
<!-- AC:END -->
