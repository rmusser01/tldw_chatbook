---
id: TASK-16254
title: Reconcile Library harnesses with current state contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 14:04'
updated_date: '2026-08-14 14:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore current-dev Library coverage by aligning minimal test harnesses with the established prompt guards, background option persistence seam, and generic ingest-panel label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canvas-scoped tests provide the prompt guard state required by current row and import-status handlers.
- [x] #2 Unmounted ingest consent tests isolate background option persistence without requiring a mounted Textual app.
- [x] #3 Keyboard focus coverage asserts the current Import behavior label while preserving structural focus evidence.
- [x] #4 Affected Library modules and the full checkpoint pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is test-fixture reconciliation with existing Library behavior and changes no production boundary.

1. Preserve the seven checkpoint failures as RED evidence and trace each production contract to its introducing commit.
2. Add only the required prompt state and background-persistence seam to the minimal test harnesses.
3. Update the structural-focus assertion to the current visible generic-panel label.
4. Run the seven nodes, affected modules, static checks, and the exact 25-file checkpoint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciled two minimal canvas stubs with current prompt selection and mutation guards, including valid prompt-row identity and the navigate-away receipt contract.
- Replaced the real worker-decorated ingest option persistence method in the unmounted consent fixture with a test seam, matching other Library unit harnesses without changing production behavior.
- Updated the structural-focus assertion to the current visible `Import behavior` label while retaining glyph and geometry checks, and removed one unused import from the touched canvas test.
- Verified the seven original failures (7 passed), the three affected modules (77 passed), Ruff lint, diff hygiene, and the exact checkpoint in two ordered chunks (314 + 503 = 817 passed; 2 existing dependency warnings). Ruff formatting remains baseline-red in all three touched modules at HEAD, so no unrelated formatting churn was introduced.
- ADR check: no ADR required; all changes are test-only reconciliation with established Library contracts.
<!-- SECTION:NOTES:END -->
