---
id: TASK-29228
title: Capture Tool policy profiles in MCP Permissions and Test Tool
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 15:13'
labels:
  - tool-packs
  - mcp
  - permissions
  - ui
dependencies:
  - TASK-29227
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make MCP Permissions and Test Tool operate on one explicitly selected local Tool policy profile, with immutable event context and stale-action rejection so profile switches can never retarget reads, edits, approvals, or executions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 MCP Permissions presents a clearly local Tool policy profile selector containing `default`, valid local, imported, and workspace-managed profiles; validated tombstones stay hidden and invalid lifecycle profiles remain visible but unavailable.
- [ ] #2 Every matrix row, inspector/re-allow action, confirmation, approval, preview, and Test Tool request carries one immutable `PermissionProfileContext` with exact profile id, selector generation, current policy digest, and imported revision where applicable.
- [ ] #3 Switching profiles or changing the selected profile's digest/revision makes older events stale; no stale event is retargeted to the current profile or mutates, approves, previews, or executes against either profile.
- [ ] #4 Global/server/tool/builtin reads and mutations, definition re-allow, persistent and session approvals, and Test Tool gates use the exact captured profile; profile-specific operations cannot change `default` or another profile, while the global kill switch remains profile-neutral.
- [ ] #5 Persistent and session mutation boundaries revalidate the captured digest/revision under the permission-store fence, and Test Tool holds an exact-profile lifecycle lease until every terminal outcome so removal cannot race an in-flight run.
- [ ] #6 Invalid/tombstoned profiles are non-editable and non-testable, selection changes clear armed confirmation and child-panel state, and stable stale/unavailable feedback leaks no policy content or sensitive diagnostics.
- [ ] #7 Focused MCP Permissions, Workbench, control-plane permission, and Test Tool suites plus scoped Ruff/format/diff checks pass; independent review has no unresolved Critical or Important findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory the current MCP Permissions canvas, Workbench action/event flow, control-plane profile-aware APIs, and Test Tool worker lifecycle; record any plan snippets that conflict with concrete interfaces.
2. Add failing tests for selector contents, immutable profile context propagation, profile-switch/digest/revision staleness, invalid lifecycle rows, and kill-switch neutrality.
3. Add the captured profile context to canvas rows/events and Workbench child requests, with generation invalidation on selection changes.
4. Add under-fence digest/revision comparison to persistent/session control-plane mutations and thread the exact captured profile through reads, edits, re-allow, preview, approvals, and Test Tool gates.
5. Hold the shared lifecycle lease for the exact captured profile through every Test Tool terminal outcome and reject stale context without substituting current selection.
6. Run the prescribed four-suite matrix, scoped static/diff checks, self-review, and independent review; remediate findings before closeout.
7. Scope correction from interface preflight: include `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` and `Tests/UI/test_mcp_inspector.py`, because `MCPInspector.ToolTestRequested` and `MCPInspector.ReallowRequested` are the concrete child-request types that must carry the captured profile context.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the selected-profile editing authority, immutable captured context, under-fence stale rejection, global kill-switch exception, and Test Tool lease boundary implemented by this task.
<!-- SECTION:PLAN:END -->
