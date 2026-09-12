---
id: TASK-31676
title: Restore Personas expression reference runtime resolution
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:12'
updated_date: '2026-09-05 18:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the expression generation failures exposed by the integrated UI sweep while preserving deferred import behavior and cancellation contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expression generation constructs the canonical resolved reference image at runtime without broadening route preimport closure.
- [x] #2 Reported expression generation failures are diagnosed and repaired without weakening generation, cancellation or first-clear assertions.
- [x] #3 Focused expression tests and relevant import/static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the integrated Personas expression-generation failures with the existing complete expression test file and trace the first failing operation.
2. Import the canonical ResolvedReferenceImage locally at its runtime construction site, preserving the TYPE_CHECKING-only route import boundary.
3. Rerun the complete expression file before diagnosing any remaining first-clear, concurrency or cancellation failures; retain all existing assertions.
4. Verify expression-slot behavior and scoped static/import checks, review and commit the bounded repair.
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: Routine missing runtime name repair that preserves existing first-use import deferral; no generation, concurrency or cancellation policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored canonical ResolvedReferenceImage resolution with a two-line first-use import at construction. The route keeps its TYPE_CHECKING-only binding; strengthened the existing 31-request/three-worker test to check exact class identity and absence of a module-global binding. No generation or cancellation policy changed.
Baseline: 9 failed, 98 passed in 314.37s (five direct NameErrors and four provider-entry assertions). After: complete generation file 107 passed in 382.87s; expression slots plus whole-registry preimport budget 49 passed in 186.69s. The independently reported first-clear cancellation race did not reproduce in either full-file run and its assertions remain unchanged. Shared fixture FD-growth warnings remain disclosed (623 and 302); no assertion or threshold suppression.
Ruff check and screen format pass; touched test-function range-format passes after normalizing one adjacent assertion, with unrelated pre-existing file formatting preserved. git diff --check and parent review passed. Existing ADR-097 applies; no new ADR or architectural boundary. Runtime-use verification is necessary for TYPE_CHECKING import deferrals; the canonical reference regression records this incident.
<!-- SECTION:NOTES:END -->
