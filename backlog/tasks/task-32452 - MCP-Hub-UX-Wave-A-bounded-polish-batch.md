---
id: TASK-32452
title: 'MCP Hub UX Wave A: bounded polish batch'
status: In Progress
assignee: []
created_date: '2026-09-11 18:34'
updated_date: '2026-09-11 18:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven bounded UI fixes from the 2026-09-11 MCP screen UX review (specs on docs/mcp-hub-ux-program): auto-focus the Permissions matrix on mode entry, ellipsize the Tools Server column, surface error reasons in permission toasts, restart markers on restart-class tool gates, one-line Advanced error rendering, width-budgeted rail legend abbreviation, and t-on-cursor-row fallback for Test Tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Permissions matrix receives focus when entering Permissions mode,Tools Server column truncates with an ellipsis never mid-word silent clip,Permission failure toast includes the service error reason,web_deep_search gate shows a restart marker; immediate gates do not,Advanced pane renders load errors as a one-line status not raw JSON,Rail legend abbreviates below the width budget with a completeness-pinned map,t opens Test Tool for the Tools-table cursor row when the inspector has no selection
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (routine bounded UI fixes; ADRs 148/149 cover the structural waves). TDD per item, targeted test runs only. 1. Test+impl: auto-focus #mcp-perm-table on set_mode('permissions'). 2. Test+impl: ellipsize Tools Server cell (fixed 24-char budget). 3. Test+impl: cycle-failure toast carries str(exc) first line via _toast. 4. Test+impl: ToolGate.restart_required + ' (⟳ restart)' label suffix on web_deep_search. 5. Test+impl: inspector renders advanced load errors as one-line status. 6. Test+impl: rail legend short-label map below width budget, completeness-pinned. 7. Test+impl: open_test_for_selected_tool falls back to Tools cursor row.
<!-- SECTION:PLAN:END -->
