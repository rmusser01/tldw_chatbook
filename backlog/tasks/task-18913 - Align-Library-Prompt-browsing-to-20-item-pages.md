---
id: TASK-18913
title: Align Library Prompt browsing to 20-item pages
status: Done
assignee: []
created_date: '2026-08-15 02:46'
updated_date: '2026-08-16 15:04'
labels:
  - library
  - pagination
  - prompts
dependencies:
  - TASK-18912
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make all Prompts reachable through the approved 20-item Library pager while preserving Prompt-specific browse, debounce, source authority, mutation history, and versioned cross-page selection behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt browse uses 20-item pages with exact applied ranges, totals, Previous, Next, loading, and recoverable error presentation.
- [x] #2 Search, sort, and Prompt collection scopes apply to the complete source before paging and successful scope changes start on page 1.
- [x] #3 The existing version-captured Prompt selection basket remains cross-page; paging or scope changes neither clear nor implicitly add entries.
- [x] #4 Prompt normalized current_page, page alias, per_page, exact total, cardinality, and stable identities are validated; malformed envelopes fail closed.
- [x] #5 An out-of-range Prompt request applies its single coherent clamped response without a redundant second service call.
- [x] #6 Prompt focus, navigation restoration, stale-generation, unmount, and dedicated-request isolation behaviors match the approved design.
- [x] #7 Automated service/state, mounted Textual, geometry, mutation, and isolated live verification pass with no regression to Prompt history or source behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: direct implementation of the accepted source-owned Prompt pagination contract; no storage, runtime, or ownership boundary changes.

Detailed plan: Docs/superpowers/plans/2026-08-15-task-18913-library-prompt-pagination.md

1. Pin the Library Prompt scope default to 20 and verify its explicit DB/service propagation without changing generic API defaults, coherent transactions, clamp, filter, or stable order.
2. Preserve and validate normalized current_page/page/per_page, exact totals/cardinality, and stable unique local identities.
3. Extend the existing Prompt browse controller with requested/applied last-good recovery state and the shared pure pager display.
4. Wire restore, debounce, focus, generations, unmount, broad-snapshot isolation, and the immutable cross-page selection basket through mounted tests.
5. Render the Prompt-specific pager at both supported geometries with retained rows, disabled reasons, and Retry.
6. Preserve delete/undo/history/version behavior and add committed-but-stale recovery.
7. Run inverse mutations, owner/live/privacy/static gates, reviews, docs, and Backlog closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-067 Prompt tranche with an explicit Library-owned 20-item
scope while preserving the generic Prompt DB/service defaults. Strict normalized
page coordinates, totals, cardinality, and stable identities now fail closed.
The existing controller owns requested scope separately from its last applied
page, retains validated rows during loading/failure, and derives the shared pager
display without a second controller or a second clamp read.

The Library screen and Prompt canvas now persist only applied scope, reject unsafe
restored offsets, isolate stale/unmounted/broad results, render exact ranges and
disabled reasons at 100x30 and 170x48, and keep the version-captured selection
basket across page and scope changes. Delete/Undo invalidate authority before the
write, reconcile known rows as committed-stale, keep row/bulk actions read-only,
and refresh the full applied scope. A live run exposed and fixed a real Textual
focus race by deferring pager-focus restoration through the loading recompose.

Key modified areas were the Prompt DB/service normalization tests, Prompt browse
state/controller, Library screen, Prompt canvas, mounted owner tests, and Library
user guides. No new ADR was required; the implementation directly follows
`backlog/decisions/067-library-top-level-pagination-contracts.md`.

Verification and review:

- Task-local DB/service/state/controller/pager gate: 475 passed.
- Final mounted Prompt owner gate: 382 passed, 73 deselected. An earlier run's
  transient history-button query was traced to state preceding DOM recompose;
  the test now uses the existing bounded selector wait, with the full Prompt
  history subset passing 57 tests.
- Required inverse mutations turned RED for the Library 20 default, coordinate
  aliases/identity validation, redundant clamp read, stale generation, applied
  restore scope, mutation page reset, stale action enablement, layout/title,
  unmount/navigation authority, and loading pager focus; every mutation was
  restored before the final gates.
- Isolated real TUI profiles at 100x30 and 170x48 each exited 0 after proving
  three exact pages over 45 synthetic rows, page/final focus, full-source search,
  sort and collection reset, a 21-item cross-page version basket, controlled
  page-3 failure and Retry, single-dispatch page-99 clamp, and detail/back restore.
  Both reported zero real-profile handles, foreign DB/config handles, TCP
  listeners, or private values in logs. The real-profile fingerprint remained
  byte-identical before and after (`25f3bbe5...`, 443 files, 289161600 bytes).
- Ruff, generated CSS parity, and diff checks passed. Independent final spec and
  quality reviews found no Critical or Important issue.

Deviation: the plan's repository-wide diagnostic command was started, then
stopped at the user's explicit direction to run only tests for modified/touched
components. No repository-wide result is claimed; the task-local and Prompt owner
gates above are the closeout authority for this tranche.
<!-- SECTION:NOTES:END -->
