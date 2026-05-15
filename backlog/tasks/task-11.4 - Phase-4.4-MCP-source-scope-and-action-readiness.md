---
id: TASK-11.4
title: 'Phase 4.4: MCP source scope and action readiness'
status: Done
assignee: []
created_date: '2026-05-12 00:00'
updated_date: '2026-05-14 19:55'
labels:
  - product-maturity
  - phase-4-agent-execution
  - mcp
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/specs/2026-04-21-unified-mcp-control-plane-parity-design.md
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make MCP source scope server hierarchy and action readiness clear while preserving the Unified MCP control plane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP presents local/server scope and server-first hierarchy without flattening tools into generic settings.
- [x] #2 User can tell which server or tool actions are ready blocked or policy-denied.
- [x] #3 Existing MCP route aliases and Unified MCP panel behavior remain compatible.
- [x] #4 QA walkthrough and focused regression evidence prove the MCP flow is usable in the running app.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression coverage for server-first MCP inventory copy and action-readiness messaging.
2. Update the Unified MCP panel so readiness state uses explicit ready, blocked, and policy-denied messages.
3. Preserve existing route aliases and mounted Unified MCP behavior while clarifying the compact destination columns.
4. Capture focused test evidence and a rendered screenshot for user approval before marking the task done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added MCP workbench regressions for server-first columns, local/server scope visibility, overview/inventory copy, ready/blocked/policy-denied action readiness, route alias compatibility, and the top navigation overflow hint. Updated the Unified MCP panel to use clear three-column Textual-native panes with divider handles for future resizing, moved `Section` near the top of the source/scope pane, hid payload editing when no action is runnable, and added explicit recovery copy for empty and policy-blocked states. Updated the MCP overview/inventory renderers so users see scannable counts and the next action before raw diagnostics. Captured and user-approved the final actual textual-web screenshot at `Docs/superpowers/qa/product-maturity/phase-4/mcp-source-scope-final-real-viewport-2026-05-14.png`; focused MCP verification passed with `16 passed` and `git diff --check` passed.
<!-- SECTION:NOTES:END -->
