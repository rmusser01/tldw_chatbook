---
id: TASK-563
title: Avoid orphaned MCP inspector clear coroutines
status: Done
assignee: []
created_date: '2026-07-25 18:53'
updated_date: '2026-07-25 18:57'
labels:
  - mcp
  - workers
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make exclusive MCP workbench transitions create inspector-clear coroutines lazily so cancellation before worker start cannot leak unawaited coroutine objects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audit Open tool and Adjust permission transitions emit no unawaited-coroutine warning under warning-strict tests.
- [x] #2 Exclusive MCP detail-disarm and inspector-clear workers use lazy callable contracts that remain safe when superseded before start.
- [x] #3 The focused transitions and full MCP workbench module pass with static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a routine worker-dispatch lifecycle repair within the existing MCP workbench boundary; no service or cross-module contract changes.

1. Reproduce the two audit deep-link warnings under a warning-strict gate.
2. Change supersedable exclusive worker dispatches to pass lazy async callables instead of already-created coroutine objects.
3. Add/strengthen regression coverage for cancellation-before-start without changing audit deep-link outcomes.
4. Run focused warning-strict tests, the full MCP workbench module, Ruff/format/diff checks, and resume TASK-546.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed MCP workbench mode-transition worker dispatch to pass lazy async callables instead of already-created coroutine objects. The detail-disarm and inspector-clear workers are now safe when an exclusive successor cancels them before start; audit Open tool and Adjust permission drill-through workers use functools.partial for the same lazy contract. Updated the explanatory contract to describe callable cancellation rather than an intentionally orphaned coroutine.

Both affected audit tests now treat RuntimeWarning as an error, permanently pinning the cancellation-before-start lifecycle. RED evidence: both failed under -W error with MCPWorkbench._clear_tool_view was never awaited. GREEN verification: focused warning-strict gate 2 passed; full Tests/UI/test_mcp_workbench.py 158 passed with zero warnings; scoped Ruff, formatter, and git diff checks passed. ADR required: no. Modified: tldw_chatbook/UI/MCP_Modules/mcp_workbench.py, Tests/UI/test_mcp_workbench.py, and this task file.
<!-- SECTION:NOTES:END -->
