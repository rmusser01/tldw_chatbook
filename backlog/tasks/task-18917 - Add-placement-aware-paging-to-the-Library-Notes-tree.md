---
id: TASK-18917
title: Add placement-aware paging to the Library Notes tree
status: In Progress
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
- [ ] #1 Every Note remains reachable through bounded placement-aware folder expansion or paging while parent-child relationships remain correct.
- [ ] #2 Folder-local totals, ranges, loading, retry, and stale states are truthful and never derived from a partial broad snapshot.
- [ ] #3 Create, move, rename, restore, delete, deep-link, and back navigation retain deterministic stable-ID placement within the tree.
- [ ] #4 Paging does not flatten the Notes hierarchy or move tree state into a generic Library controller.
- [ ] #5 Keyboard focus, expansion state, narrow-terminal geometry, request races, unmount behavior, and recoverable failures have mounted regression coverage.
- [ ] #6 Automated tree/service/state tests and isolated live verification with a large synthetic hierarchy pass.
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
