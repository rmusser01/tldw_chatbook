---
id: TASK-34915
title: Gate-the-MCP-save-status-config-file-stat-tick
status: To Do
assignee: []
created_date: '2026-10-13 14:45'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP workbench arms set_interval(0.25, _sync_local_config_save_status) (mcp_workbench.py:986); the tick builds its dirty-check stamp from current_config_identity() plus config_file_revision(get_cli_config_path()) — a config-file identity/stat read 4x/second plus two DOM query() calls for as long as the MCP screen is open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-tick work no longer stats or re-reads the config file,Revision checked on change events or a coarser cadence,Save-status UI still updates correctly when the file changes (test)
<!-- AC:END -->
