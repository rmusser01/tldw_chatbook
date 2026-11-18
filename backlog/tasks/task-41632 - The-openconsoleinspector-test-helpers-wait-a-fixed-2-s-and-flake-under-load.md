---
id: TASK-41632
title: "The `_open_console_inspector` test helpers wait a fixed 2 s and flake under load"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- flaky
- console
priority: low
references:
- Tests/UI/test_console_internals_decomposition.py
- Tests/UI/test_console_scope_row.py
dependencies:
- TASK-41308
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41308's MCP-row tests open the Inspector through `_open_console_inspector` (`Tests/UI/test_console_internals_decomposition.py:275`; a copy is at `Tests/UI/test_console_scope_row.py:141`). The helper gives the rail a fixed 2 s deadline, and it timed out in local runs at load 50-100 when the same tests passed alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both helpers wait on the Inspector's visible condition with a deadline that tolerates a loaded runner (or the shared wait helper), not a fixed 2 s
- [x] #2 The MCP-row tests pass in a `-n 4` run of their files
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test-helper timing only)

1. Keep both helpers polling the Inspector's visible condition; replace the fixed 2 s with a `timeout` keyword defaulting to a load-tolerant 15 s, and pass it to the selector wait the decomposition helper does first (that wait also defaulted to 2 s).
2. Run both files at `-n 4` under the shared machine's load and confirm the MCP-row tests pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Change.** Both `_open_console_inspector` helpers (`Tests/UI/test_console_internals_decomposition.py`, and its copy in `Tests/UI/test_console_scope_row.py`) take `timeout: float = 15.0` instead of a fixed 2 s. They still poll the visible condition (rail displayed with width, and in the decomposition copy the run-inspector state laid out) and return as soon as it holds, so the longer deadline costs nothing on a pass and only bounds a real failure. The decomposition helper's first wait, `_wait_for_selector(..., "#console-inspector-rail-open")`, now gets the same deadline; its 2 s default was the second fixed wait on that path. No shared load-scaled wait helper exists in `Tests/UI`, so none was added; `Tests/UI/test_console_send_without_mcp_store.py` and `Tests/UI/test_console_workbench_contract.py` import the decomposition helper and get the change too.

**Evidence** (worktree fu2-redui at d0486978dd, scratch-profile wrapper):
- Both files together at `-n 4`, load 95 at start: the five MCP-row tests (`test_console_run_inspector_shows_mcp_tools_ready_row`, `..._shows_mcp_not_connected_row`, `..._omits_mcp_row_by_default`, `..._mcp_row_reflects_real_compose_mcp_provider`, `..._mcp_row_shows_blocked_when_stale_server_has_tools`) all passed. In that run the two files had 162 other failures, nearly all `RecoveryRequired: raw_source_selection_changed` on tests without the `bootstrap_profile` marker; TASK-41633 fixes those.

Files: `Tests/UI/test_console_internals_decomposition.py`, `Tests/UI/test_console_scope_row.py`.
<!-- SECTION:NOTES:END -->
