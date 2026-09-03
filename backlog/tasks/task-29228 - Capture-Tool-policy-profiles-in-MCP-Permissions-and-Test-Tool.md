---
id: TASK-29228
title: Capture Tool policy profiles in MCP Permissions and Test Tool
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 15:13'
updated_date: '2026-09-02 20:10'
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
- [x] #1 MCP Permissions presents a clearly local Tool policy profile selector containing `default`, valid local, imported, and workspace-managed profiles; validated tombstones stay hidden and invalid lifecycle profiles remain visible but unavailable.
- [x] #2 Every matrix row, inspector/re-allow action, confirmation, approval, preview, and Test Tool request carries one immutable `PermissionProfileContext` with exact profile id, selector generation, current policy digest, and imported revision where applicable.
- [x] #3 Switching profiles or changing the selected profile's digest/revision makes older events stale; no stale event is retargeted to the current profile or mutates, approves, previews, or executes against either profile.
- [x] #4 Global/server/tool/builtin reads and mutations, definition re-allow, persistent and session approvals, and Test Tool gates use the exact captured profile; profile-specific operations cannot change `default` or another profile, while the global kill switch remains profile-neutral.
- [x] #5 Persistent and session mutation boundaries revalidate the captured digest/revision under the permission-store fence, and Test Tool holds an exact-profile lifecycle lease until every terminal outcome so removal cannot race an in-flight run.
- [x] #6 Invalid/tombstoned profiles are non-editable and non-testable, selection changes clear armed confirmation and child-panel state, and stable stale/unavailable feedback leaks no policy content or sensitive diagnostics.
- [x] #7 Focused MCP Permissions, Workbench, control-plane permission, and Test Tool suites plus scoped Ruff/format/diff checks pass; independent review has no unresolved Critical or Important findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory the current MCP Permissions canvas, Workbench action/event flow, control-plane profile-aware APIs, and Test Tool worker lifecycle; record any plan snippets that conflict with concrete interfaces.
2. Add failing tests for selector contents, immutable profile context propagation, profile-switch/digest/revision staleness, invalid lifecycle rows, and kill-switch neutrality.
3. Add the captured profile context to canvas rows/events and Workbench child requests, with generation invalidation on selection changes.
4. Add under-fence digest/revision comparison to persistent/session control-plane mutations and thread the exact captured profile through reads, edits, re-allow, preview, approvals, and Test Tool gates.
5. Hold the shared lifecycle lease for the exact captured profile through every Test Tool terminal outcome and reject stale context without substituting current selection.
6. Run the prescribed four-suite matrix, scoped static/diff checks, self-review, and independent review; remediate findings before closeout.
7. Scope correction from interface preflight: include tldw_chatbook/UI/MCP_Modules/mcp_inspector.py and Tests/UI/test_mcp_inspector.py, because the inspector request types are the concrete child requests that must carry captured profile context.
8. Review remediation: expose the application ToolPack lifecycle coordinator, render rows and profile context from one fenced identity, revalidate queued tests and persistent-approval successors exactly, fail closed on corrupt profile inventory, and preserve legacy hash-free Allow entries while keeping new writes canonical.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the selected-profile editing authority, immutable captured context, under-fence stale rejection, global kill-switch exception, and Test Tool lease boundary implemented by this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented explicit Tool policy profile selection and immutable PermissionProfileContext propagation across Permissions rows, inspector actions, approvals, previews, and Test Tool requests.

Added exact-profile control-plane reads and fenced mutations, stale digest/revision rejection, coherent fenced render snapshots, production ToolPack lifecycle lease wiring, and exact pre-execution gate revalidation. Invalid and tombstoned profiles fail closed; corrupt authority no longer synthesizes a usable default. Legacy hash-free Allow entries with an explicit null hash remain readable, while new hash-free writes omit the field and other Allow hashes require SHA-256.

Verification: 280 Workbench tests, 106 permission-store/control-plane permission tests, and 295 Permissions mode/Inspector/Test Tool/ToolPack service tests passed. Scoped Ruff lint, changed-range Ruff format, and full branch git diff checks passed. The independent review reported no unresolved Critical or Important findings and a ready verdict. The repository-wide suite was not run, per the targeted-test policy.

ADR required: no new ADR. Existing ADR: backlog/decisions/107-portable-tool-use-packs.md.

Primary files: MCP permission store and control-plane service; ToolPack service; MCP Permissions, Inspector, and Workbench UI modules; focused unit and integration tests.

Latest-dev rebase follow-up: ported the profile context and lifecycle lease onto
dev's service-owned prepared Test Tool flow. Prepared nonces now bind the exact
profile id, policy digest, and imported revision; preview minting rejects stale or
invalid lifecycle authority, click-time policy drift returns a bounded stale result,
and local provider re-resolution uses the captured profile through dispatch. The UI
revokes rather than arms refreshed authority after a profile change. Rebase fixtures
were also aligned with the hardened SHA-256 definition-hash contract.

Latest-dev verification: 568 MCP permission/store/control-plane/Inspector/ToolPack
tests and all 332 Workbench tests passed. The rebased Settings/profile checks added
33 passing UI tests, and the V1 closure checks added 76 passing Tool Pack/Console
continuation tests. Scoped Ruff lint/format and git diff checks passed; the full
repository suite was not run under the targeted-test policy.
<!-- SECTION:NOTES:END -->
