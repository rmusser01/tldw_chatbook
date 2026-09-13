---
id: TASK-31653
title: Reconcile Chat fixture and MCP exclusion contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:02'
updated_date: '2026-09-05 17:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Chat tests whose fixture signatures and fixed catalog counts no longer match current production contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The promotion fake accepts absent trace boundaries but refuses unsupported nonempty trace persistence.
- [x] #2 The MCP test verifies exact descriptor plus legacy exclusions without pinning an obsolete total.
- [x] #3 Affected Chat tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce current trace_boundary keyword and MCP count failures and inspect production contracts.
2. Update the minimal persistence fake signature and retain an explicit unsupported-boundary refusal.
3. Remove the redundant stale exclusion count while preserving exact set and composed-provider assertions.
4. Run affected store/controller tests, scoped static checks, and review.
ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md (existing)
Reason: test-only alignment with existing promotion and catalog contracts, no runtime behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the FakePersistence promotion signature with the optional trace boundary, while explicitly refusing nonempty boundaries rather than pretending to persist lineage. Added a red/green refusal regression. Removed the obsolete MCP total29 assertion; exact descriptor-plus-five-legacy membership and composed-provider exclusion assertions remain. All650 tests in both affected Chat store/controller files passed67.65s in the installed review environment. Ruff, changed-range formatting, diff check, and self-review passed. No runtime changes; ADR097 unchanged.
<!-- SECTION:NOTES:END -->
