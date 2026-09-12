---
id: TASK-32452
title: 'MCP Hub UX Wave A: bounded polish batch'
status: Done
assignee: []
created_date: '2026-09-11 18:34'
updated_date: '2026-09-11 19:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven bounded UI fixes from the 2026-09-11 MCP screen UX review (specs on docs/mcp-hub-ux-program): auto-focus the Permissions matrix on mode entry, ellipsize the Tools Server column, surface error reasons in permission toasts, restart markers on restart-class tool gates, one-line Advanced error rendering, width-budgeted rail legend abbreviation, and t-on-cursor-row fallback for Test Tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Permissions matrix receives focus when entering Permissions mode,Tools Server column truncates with an ellipsis never mid-word silent clip,Permission failure toast includes the service error reason,web_deep_search gate shows a restart marker; immediate gates do not,Advanced pane renders load errors as a one-line status not raw JSON,Rail legend abbreviates below the width budget with a completeness-pinned map,t opens Test Tool for the Tools-table cursor row when the inspector has no selection
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (routine bounded UI fixes; ADRs 148/149 cover the structural waves). TDD per item, targeted test runs only. 1. Test+impl: auto-focus #mcp-perm-table on set_mode('permissions'). 2. Test+impl: ellipsize Tools Server cell (fixed 24-char budget). 3. Test+impl: cycle-failure toast carries str(exc) first line via _toast. 4. Test+impl: ToolGate.restart_required + ' (⟳ restart)' label suffix on web_deep_search. 5. Test+impl: inspector renders advanced load errors as one-line status. 6. Test+impl: rail legend short-label map below width budget, completeness-pinned. 7. Test+impl: open_test_for_selected_tool falls back to Tools cursor row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: TDD per item (test watched RED, then minimal implementation), all seven items on `feat/mcp-hub-ux-wave-a` (commit 8a93ab5).
- A1 focus: `set_mode("permissions")` focuses `#mcp-perm-table` via `call_after_refresh`; focus already inside the permissions canvas (the filter Input) is respected. A `t`-fallback regression surfaced mid-wave: the first draft fired from any mode (a F-055 mode-hijack), fixed by gating the fallback on `active_mode == "tools"` — the pre-existing no-selection test caught it.
- A2: fixed 24-column budget + explicit `…` (DataTable has no cell ellipsis/tooltips; width-aware refinement deferred — the server-filter Select still shows full labels).
- A3: toast carries the first line of the service error (≤140 chars), escaped via `_toast`; stale-profile branch copy unchanged.
- A4: `ToolGate.restart_required` (default False; True only for `web_deep_search`) + `(⟳ restart)` label suffix; save/reload path untouched (id still built from `gate.key`).
- A5: `_render_section_payload` branches on a truthy `error` key (the `_AdvancedSectionShim` failure shape) → one-line status; empty error still renders JSON.
- A6: `_SHORT_STATE_LABELS` keyed off `STATE_LABELS` values with a completeness-pinned test; budget hoisted above the legend yield in `MCPRail.compose` so rows and legend share one computation.
- A7: `_open_test_for_tools_cursor_row` resolves the cursor row and drives the same `show_tool` path a selection takes, then retries `open_test_panel`.
- ADR check: none required (routine bounded UI fixes; ADRs 148/149 cover the structural waves). Historical phase docs (2026-07 specs/QA) intentionally left untouched.
- Verification: targeted suites green — test_mcp_rail 25, test_mcp_tools_mode 30, test_mcp_servers_mode 60, test_mcp_inspector 283, test_mcp_workbench 337 (final combined A+B run: 515 passed). Ruff: no new findings vs clean HEAD. Pre-existing unrelated failures in Tests/MCP/test_mcp_documentation_contract.py (39) exist on clean origin/dev — verified via stash; not addressed here.
<!-- SECTION:NOTES:END -->
