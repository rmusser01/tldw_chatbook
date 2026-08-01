---
id: TASK-1652
title: Reconcile concurrent Backlog task ID collisions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 00:50'
updated_date: '2026-08-01 01:00'
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
- [x] #2 Later colliding Settings tasks use the approved TASK-1660 through TASK-1665 mapping.
- [x] #3 Every task frontmatter ID is uppercase and unique.
- [x] #4 All in-repository references identify the intended renamed task.
- [x] #5 The Backlog identity sentinel and diff hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is task-ledger and reference hygiene; it does not change a runtime, storage, privacy, or application architecture boundary.

1. Record the deterministic older-wins mapping from commit creation order.
2. Rename the six later Settings task files and update their frontmatter IDs.
3. Uppercase every remaining lowercase task frontmatter ID without changing its number.
4. Update only references belonging to the six renamed Settings tasks; preserve Console and Library references to older IDs.
5. Run the Backlog identity sentinel, targeted searches, formatting/diff hygiene, and self-review.
6. Record implementation evidence and mark TASK-1652 Done only after all gates pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled concurrent IDs with the approved older-wins mapping: Settings TASK-1620/1621/1622 became TASK-1660/1661/1662 and Settings TASK-1640/1641/1642 became TASK-1663/1664/1665. Uppercased the remaining lowercase frontmatter IDs, updated only Settings-owned code, test, CSS, and design references, and preserved the older Console and Library references. ADR required: no (ledger/reference hygiene only). Verification: the Backlog identity sentinel passed; the complete product-maturity harness passed; an independent parser validated 1,150 task files with zero malformed IDs, filename mismatches, or duplicates; scoped searches and git diff --check passed.

Static analysis: Ruff lint passed. Ruff format still reports the same pre-existing drift in three files; an in-memory comparison proved every touched Python file differs from HEAD only by the approved task-ID mapping and that its formatter output is baseline-equivalent. No unrelated reformatting was introduced.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked.
- [x] #2 The approved mapping and affected references are documented.
- [x] #3 Automated identity and focused reference tests pass.
- [x] #4 Self-review and diff hygiene pass.
<!-- DOD:END -->
