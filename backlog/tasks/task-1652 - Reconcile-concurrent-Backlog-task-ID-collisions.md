---
id: TASK-1652
title: Reconcile concurrent Backlog task ID collisions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 00:50'
updated_date: '2026-08-01 06:56'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore one canonical uppercase TASK-* identity per Backlog task after concurrent dev branches allocated overlapping numbers, so repository task references and the task-hygiene sentinel are reliable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Earlier-created tasks retain their existing numeric IDs.
- [x] #2 Canonical dev TASK-1660/TASK-1661 retain their IDs and the colliding Settings tasks use the approved TASK-1711 through TASK-1716 mapping.
- [x] #3 Every task frontmatter ID is uppercase and unique.
- [x] #4 All in-repository references identify the intended renamed task.
- [x] #5 The Backlog identity sentinel and diff hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is task-ledger and reference hygiene; it does not change a runtime, storage, privacy, or application architecture boundary.

1. Preserve canonical dev TASK-1660/TASK-1661 and allocate the next free contiguous range, TASK-1711 through TASK-1716, to the six colliding branch-only Settings tasks.
2. Rename the six Settings task files and update their frontmatter IDs.
3. Uppercase every remaining lowercase task frontmatter ID without changing its number.
4. Update only references belonging to the six renamed Settings tasks; preserve Console and Library references to older IDs.
5. Run the Backlog identity sentinel, targeted searches, formatting/diff hygiene, and self-review.
6. Record implementation evidence and mark TASK-1652 Done only after all gates pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The initial reconciliation moved the six colliding Settings tasks out of TASK-1620/1621/1622 and TASK-1640/1641/1642. A later rebase then brought canonical dev TASK-1660 (graphics protocol) and TASK-1661 (rail avatar) into the range used by that first pass, so TASK-1652 was reopened. The canonical dev tasks retained their merged IDs; the six branch-only Settings tasks moved together to the next free contiguous range, TASK-1711 through TASK-1716. Only Settings-owned code, tests, CSS, and design references changed; graphics/avatar TASK-1660/TASK-1661 references were preserved. Both newly landed canonical frontmatter IDs were normalized to uppercase.

ADR required: no (ledger/reference hygiene only). Verification: the Backlog identity sentinel passed; the complete product-maturity harness passed (`2 passed`); an independent numeric-segment parser validated 1,164 task files with 1,164 unique filename/frontmatter identities and zero malformed IDs, mismatches, or duplicates; scoped searches and `git diff --check` passed. Ruff lint passed. Ruff format still reports the same pre-existing drift in three files; the touched Python diffs are comment-only task-reference substitutions, so no unrelated formatter churn was introduced.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked.
- [x] #2 The approved mapping and affected references are documented.
- [x] #3 Automated identity and focused reference tests pass.
- [x] #4 Self-review and diff hygiene pass.
<!-- DOD:END -->
