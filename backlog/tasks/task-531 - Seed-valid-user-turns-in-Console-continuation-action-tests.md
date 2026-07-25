---
id: TASK-531
title: Seed valid user turns in Console continuation action tests
status: Done
assignee: []
created_date: '2026-07-24 19:38'
updated_date: '2026-07-24 19:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Continue and Regenerate UI coverage compatible with the guard that rejects assistant-only transcripts before the first user message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Continue and Regenerate action fixtures include a real user turn before the selected assistant response
- [x] #2 The tests still verify streamed continuation and persisted sibling-regeneration behavior
- [x] #3 The focused action tests and full native Console flow module pass
- [x] #4 The guard source and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the assistant-only fixture failures and trace the intended user-turn guard to commit 38380380c.
2. Seed a user parent before each selected assistant response without changing production logic.
3. Run the focused Continue/Regenerate tests, full native Console flow module, Ruff, format, and diff checks.
4. Independently review the fixture repair and document verification.

ADR required: no
ADR path: N/A
Reason: This corrects stale test setup for an existing controller safety contract and changes no production interface or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Seeded a user message before the selected assistant response in the Continue and Regenerate UI tests. This gives both actions the valid user→assistant history required by the controller while preserving assertions for streamed continuation output, source immutability, active-path replacement, and persisted sibling-node generation.

Commit `38380380c` added the intentional guard that rejects Continue/Regenerate on assistant-only transcripts, but these older fixtures still constructed that invalid state. The focused regression cases pass 4/4 and the complete native Console flow module passes 199/199.

ADR required: no. This is test-fixture alignment to an existing controller safety contract and changes no production boundary.
<!-- SECTION:NOTES:END -->
