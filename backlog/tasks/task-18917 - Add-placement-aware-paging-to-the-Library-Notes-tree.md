---
id: TASK-18917
title: Add placement-aware paging to the Library Notes tree
status: Done
assignee: []
created_date: '2026-08-15 02:50'
labels:
  - library
  - pagination
  - notes
  - follow-up
dependencies:
  - TASK-18912
  - TASK-18913
  - TASK-18914
  - TASK-18915
  - TASK-18916
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - >-
    Docs/superpowers/specs/2026-08-29-task-18917-library-notes-tree-placement-aware-paging-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make large Notes folder trees fully reachable without flattening parent-child relationships or replacing the existing tree-specific Load-more ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Note remains reachable through bounded placement-aware folder expansion or paging while parent-child relationships remain correct.
- [x] #2 Folder-local totals, ranges, loading, retry, and stale states are truthful and never derived from a partial broad snapshot.
- [x] #3 Create, move, rename, restore, delete, deep-link, and back navigation retain deterministic stable-ID placement within the tree.
- [x] #4 Paging does not flatten the Notes hierarchy or move tree state into a generic Library controller.
- [x] #5 Keyboard focus, expansion state, narrow-terminal geometry, request races, unmount behavior, and recoverable failures have mounted regression coverage.
- [x] #6 Automated tree/service/state tests and isolated live verification with a large synthetic hierarchy pass.
<!-- AC:END -->

## Implementation Plan

Detailed plan:
`Docs/superpowers/plans/2026-08-29-task-18917-library-notes-tree-placement-aware-paging.md`

1. Add typed folder/placement page contracts and a pure Notes branch reducer.
2. Add exact parent-scoped child-folder and visible-placement repository pages.
3. Add exact locators, mutation context lookup, and coherent filter placement pages.
4. Expose policy-checked off-loop `NotesScopeService` seams.
5. Add the paged tree projection, inline controls, and narrow-width styling while
   preserving a runnable compatibility path.
6. Cut `LibraryScreen` browse orchestration over to branch-local workers and lifecycle
   authority.
7. Reconcile deep links, Back receipts, filters, and committed mutations, then remove
   compatibility state.
8. Run targeted production-shaped cross-reader and isolated real-repository live
   verification at 160×50, 120×35, 100×30, and 80×24; update documentation and task
   evidence.

ADR required: no

ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`

Reason: ADR-067 already governs source-owned paging, exact totals, stable locators,
generation fencing, cross-visit scope-only persistence, and this Notes hierarchy
follow-up. No storage, ownership, sync, security, dependency, or application-level
boundary changes.

## Implementation Notes

Implemented ADR-067's Notes hierarchy follow-up with exact 20-item child-folder and
visible-placement pages, branch-local state/workers, stable placement locators,
mutation reconciliation, bounded filtering, and source-owned More/Earlier/Retry
controls. Production-shaped mounted coverage verifies focus, containment, scroll,
collapse behavior, and unchanged cross-reader contracts at 160×50, 120×35,
100×30, and 80×24; evidence-driven layout corrections prevent horizontal
clipping, preserve explicit pane choices, reject stale allocations, and keep
same-ID focus mounted across Prompt/Skills recomposes. The exact targeted suite
passed 697 tests, and the isolated real ChaChaNotes/repository/service walkthrough
passed with 25 roots, 25 Unfiled notes, 25 children, 45 visible placements, deep,
duplicate, shadowed-managed, mutation, located-middle, Earlier, failure/Retry, and
all-size coverage. User documentation and
`Docs/superpowers/reviews/evidence/task-18917/live-walkthrough.md` record the
behavior and verification. No new ADR was required; ADR-067 remains authoritative.
