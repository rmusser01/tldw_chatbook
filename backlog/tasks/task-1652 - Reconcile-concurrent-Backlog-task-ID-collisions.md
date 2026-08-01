---
id: TASK-1652
title: Reconcile concurrent Backlog task ID collisions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 00:50'
updated_date: '2026-08-01 10:54'
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
- [x] #2 Canonical File Notes TASK-1711 and Settings TASK-1712 retain their IDs; the four later duplicate claimants use TASK-1717 through TASK-1720 in creation order.
- [x] #3 Every task frontmatter ID is uppercase and unique.
- [x] #4 All in-repository references identify the intended renamed task.
- [x] #5 The Backlog identity sentinel and diff hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is task-ledger and reference hygiene; it does not change a runtime, storage, privacy, or application architecture boundary.

1. Preserve the earlier canonical claimant at each duplicate ID: File Notes keeps TASK-1711 and Settings filter/overflow keeps TASK-1712.
2. Allocate the next free IDs in claimant-creation order: Settings persistence TASK-1717, Briefings transaction residual TASK-1718, Briefings coverage residual TASK-1719, and artifact source fingerprint TASK-1720.
3. Rename only the four duplicate task files and update their frontmatter IDs.
4. Update references according to domain ownership; preserve all canonical File Notes TASK-1711 and Settings TASK-1712 references plus unrelated Settings TASK-1713 through TASK-1716.
5. Run the Backlog identity sentinel, the complete product-maturity harness, an independent filename/frontmatter parser, targeted searches, lint, and diff hygiene.
6. Record implementation evidence and mark TASK-1652 Done only after all gates pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The initial reconciliation moved the six colliding Settings tasks out of TASK-1620/1621/1622 and TASK-1640/1641/1642. A later rebase then brought canonical dev TASK-1660 (graphics protocol) and TASK-1661 (rail avatar) into the range used by that first pass, so TASK-1652 was reopened. The canonical dev tasks retained their merged IDs; the six branch-only Settings tasks moved together to the next free contiguous range, TASK-1711 through TASK-1716. Only Settings-owned code, tests, CSS, and design references changed; graphics/avatar TASK-1660/TASK-1661 references were preserved. Both newly landed canonical frontmatter IDs were normalized to uppercase.

ADR required: no (ledger/reference hygiene only). Verification: the Backlog identity sentinel passed; the complete product-maturity harness passed (`2 passed`); an independent numeric-segment parser validated 1,164 task files with 1,164 unique filename/frontmatter identities and zero malformed IDs, mismatches, or duplicates; scoped searches and `git diff --check` passed. Ruff lint passed. Ruff format still reports the same pre-existing drift in three files; the touched Python diffs are comment-only task-reference substitutions, so no unrelated formatter churn was introduced.

A subsequent rebase exposed two new three-way collisions created by independently merged branches. Commit history established the canonical earlier claimants: File Notes retains TASK-1711 and Settings filter/overflow retains TASK-1712. The later tasks moved in creation order: Settings persistence to TASK-1717, Briefings transaction residual to TASK-1718, Briefings coverage residual to TASK-1719, and artifact source fingerprint to TASK-1720. References were updated only in their owning Settings, Briefings, or Model Artifacts files. Backlog CLI resolves all six retained/new IDs to the intended files. Verification: the complete product-maturity harness passed (`2 passed`); an independent dotted-numeric filename/frontmatter parser validated 1,173 task files with 1,173 unique matching IDs and no malformed or duplicate identities; scoped reference review, Ruff lint, and `git diff --check` passed. Ruff format reports inherited upstream drift in seven comment-only-touched Python files; their diffs contain task-reference substitutions only, so they were not mechanically reformatted.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked.
- [x] #2 The approved mapping and affected references are documented.
- [x] #3 Automated identity and focused reference tests pass.
- [x] #4 Self-review and diff hygiene pass.
<!-- DOD:END -->
