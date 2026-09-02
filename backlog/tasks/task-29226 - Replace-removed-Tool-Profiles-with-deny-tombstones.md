---
id: TASK-29226
title: Replace removed Tool Profiles with deny tombstones
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 06:38'
updated_date: '2026-09-02 13:55'
labels:
  - tool-packs
  - removal
  - tombstones
  - security
dependencies:
  - TASK-28225
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace removable imported Tool Profiles with permanent fail-closed Deny tombstones while active or archived references and exact-profile runtime leases block removal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Removal accepts only a valid imported profile at the expected revision with zero active or archived workspace references and zero exact-profile runtime leases; default, local/legacy, ws-, invalid, referenced, leased, and already tombstoned profiles fail closed with stable categories.
- [x] #2 Confirmed removal durably stages a new compact receipt and atomically replaces authority with a permanent hidden tombstone that preserves compact provenance, increments revision, carries a current policy digest, denies MCP-global and builtin authority, contains no Allow/Ask rows, reserves the id, and continues to count toward caps.
- [x] #3 Post-replace reconciliation treats the exact tombstone as success, the exact prior imported state as a known failure, and any other state as outcome_uncertain while retaining both old and new receipts; the prior detailed receipt becomes orphan-cleanup eligible only after proven replacement.
- [x] #4 Console runs and MCP Test Tool acquire the exact captured profile lease before governed admission/gating, hold it through the final invocation/result/error, and release it in finally; separate profile ids do not interfere and no claim of revoking dispatched work is made.
- [x] #5 Bind/removal, lease/removal, active/archived/dangling-reference, resolver-tombstone, and runtime release races are covered by deterministic targeted tests; related permission, Console, and Workbench tests plus scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing removal eligibility, tombstone authority, receipt ordering/reconciliation, reference, lease, and deterministic race tests.
2. Extend lifecycle coordination with exact-profile lease handles and wire Console/Test Tool captured-profile execution scopes through them.
3. Implement compact-receipt-first removal under lifecycle then permission-store fencing with strict reference/lease/revision revalidation.
4. Replace valid imported authority with a permanent Deny tombstone and reconcile exact known-success, known-failure, and uncertain outcomes without deleting evidence.
5. Run focused removal/runtime/resolver tests, related authority regressions, scoped Ruff/format and diff hygiene, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the runtime lease, reference authority, compact receipt, permanent tombstone, resolver short-circuit, cap accounting, and uncertain-removal contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fail-closed imported Tool Profile removal with exact runtime leases,
compact-receipt-backed permanent Deny tombstones, and strict known/uncertain
replacement reconciliation. Console and MCP Test Tool now hold the captured
profile lease through final outcomes; a task-completion backstop also releases a
Test Tool lease when Textual cancels its worker before coroutine entry.

Targeted evidence: 26 removal/resolver tests and 5 Console/Workbench lease tests
passed after the review fix. The prescribed broader matrix completed with 566
passes and 11 pre-existing Console failures that were mutation-checked against the
unmodified lease boundary. Scoped Ruff lint, new-file format checks, and diff
hygiene passed. Independent review found one Important pre-start cancellation
case; commit `10754338c` fixed it and re-review passed with no remaining findings.

Modified files: `tldw_chatbook/Tool_Packs/binding.py`,
`tldw_chatbook/Tool_Packs/removal.py`,
`tldw_chatbook/Chat/console_chat_controller.py`,
`tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`, and their focused Tool Packs,
Console, Workbench, and permission-resolution tests. ADR-107 remains controlling;
no new ADR or generalized lesson was required.
<!-- SECTION:NOTES:END -->
