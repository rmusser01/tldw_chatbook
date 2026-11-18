---
id: TASK-34944
title: Gate-the-MCP-save-status-config-file-stat-tick
status: Done
assignee: []
created_date: 2026-10-13 14:45
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: medium
updated_date: 2026-10-13 16:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP workbench arms set_interval(0.25, _sync_local_config_save_status) (mcp_workbench.py:986); the tick builds its dirty-check stamp from current_config_identity() plus config_file_revision(get_cli_config_path()) — a config-file identity/stat read 4x/second plus two DOM query() calls for as long as the MCP screen is open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Per-tick work no longer stats or re-reads the config file
- [x] Revision checked on change events or a coarser cadence
- [x] Save-status UI still updates correctly when the file changes (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _sync_local_config_save_status + config_file_revision/current_config_identity
2. Coarsen or event-drive the config-file identity read per tick
3. Unit-test the gate
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added `_save_status_file_identity(refresh=...)` on MCPWorkbench (mcp_workbench.py): caches the (current_config_identity, config_file_revision) pair. The 0.25s `_sync_local_config_save_status` timer refreshes that cache only every 8th tick (~2s) — external config edits are still detected automatically, just on a 2s instead of 0.25s cadence. Idle ticks now build their stamps purely from in-memory state (owner save revisions, completion revision) with no stat and no fresh identity walk.
- Direct callers keep fresh reads: `_sync_root_save_status` / `_sync_master_save_status` default to `file_identity_refresh=True`, so every save observer and event path (which run exactly when the file just changed) stats immediately. `_master_save_projection` takes the flag through from its callers.
- The pre-existing stamp gates (`_root_save_projection` etc.) are unchanged — they now compare against the coarser-cadence identity.
- Tests: new `Tests/UI/test_mcp_save_status_tick_gate.py` (3 tests: stat-once-until-forced cache, ticks 1-and-9-only refresh cadence, direct-call defaults). 3/3 pass. Regression: `Tests/UI/test_mcp_workbench.py` + `test_mcp_workbench_lifetime.py` — 361/361 passed (suite is green at HEAD, so a true regression pass).
- ADR required: no — cadence change inside one widget's timer, detection semantics preserved.
