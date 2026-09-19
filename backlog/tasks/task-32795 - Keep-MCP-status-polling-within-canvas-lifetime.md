---
id: TASK-32795
title: Keep MCP status polling within canvas lifetime
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 16:12'
updated_date: '2026-09-18 16:29'
labels:
  - mcp
  - ui
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent background local-setting status projection from touching incomplete or closing MCP canvases, and make startup evidence wait for the actual load boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deferred canvas construction and subtree teardown cannot cause status polling to crash the Workbench.
- [x] #2 Mounted Tools and Servers controls still receive current root and master receipts, including after a fresh mount.
- [x] #3 Startup checks wait for completed loading and still detect missing or incorrect server rows.
- [x] #4 Targeted lifecycle and adjacent regressions pass; evidence and the draft PR are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce partial canvas mount and teardown polling with deterministic barriers; inspect the initial-load test ordering. 2. Apply minimal lifecycle guards under the existing app-owned save contract, and wait for actual startup completion in its regression. 3. Verify delayed startup, prune/remount and mounted receipt projection with targeted tests and a private native journey. 4. Review the change and update the MCP ledger and draft PR2707. ADR required: no. ADR path: backlog/decisions/169-mcp-local-config-save-lifetime.md. Reason: Routine presentation-lifetime repair of the accepted save owner, without changing persistence, authority, or interaction contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Status projection now requires live mounted Workbench/canvas controls and skips pruning subtrees. Root receipt stamps include canvas identity and are consumed only after projection is possible, preserving already displayed and newly arriving receipts through replacement. Startup evidence waits for the synchronously claimed load boundary; an adjacent save test drains its scheduled render before teardown. Existing ADR-169 applies; no persistence, authority, token or stylesheet changes. Validation: 79 distinct targeted passing cases across a 60-case scoped run and final 28-case root/lifecycle replay; independent review found and verified the canvas-stamp correction. Eight final dark/light compact/wide native captures inspected, real save/navigation receipts preserved, exit 0, eleven healthy private databases and unchanged default files/policies. Ruff has no introduced findings; changed functions/new files formatted, Backlog and diagnostic guards pass. QA receipt and failed-attempt disposition: Docs/superpowers/qa/2026-09-18-mcp-workbench-lifetime/README.md. Updated MCP/completion ledgers and the mount-lifetime lesson. Two separate inspector-clearing CI cases still fail locally and remain the next review scope; no green PR or full-workstream completion claim. PR2707 remains draft for separate visual review and merge approval.
<!-- SECTION:NOTES:END -->
