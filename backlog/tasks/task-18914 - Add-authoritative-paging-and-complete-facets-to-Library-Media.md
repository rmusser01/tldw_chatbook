---
id: TASK-18914
title: Add authoritative paging and complete facets to Library Media
status: Done
assignee: []
created_date: '2026-08-15 02:47'
updated_date: '2026-08-20 08:38'
labels:
  - library
  - pagination
  - media
  - privacy
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
Make all top-level Media items reachable through bounded 20-item pages and complete type facets without prefix-fetching or exposing private query, title, identifier, or path data in diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media filtering and type selection apply to the complete source before coherent exact-total database-level 20-row offset paging.
- [x] #2 Deep-page reads are bounded and deterministic, with stable Media ID tie-breaking and exact validated envelope cardinality and identities.
- [x] #3 The Media type chooser exposes the complete distinct type set in one bounded keyboard-accessible OptionList-style control with active and cancel behavior.
- [x] #4 Media current-page selection clears with visible notice on page, filter, sort, or type scope change, and stale mutation recovery disables unsafe actions.
- [x] #5 Touched Media read diagnostics contain metadata only and never query text, titles, bodies, stable private IDs, filesystem paths, or credentials on success or error.
- [x] #6 Media request generations, unmount fencing, broad-snapshot isolation, focus, restoration, concurrent shrink, and recoverable error behavior match the approved design.
- [x] #7 Automated database/service/state, mounted Textual, geometry, privacy, mutation, and isolated live verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: Direct implementation of ADR-067's approved Media paging, facet, stale-recovery, and privacy contracts.

Detailed plan: Docs/superpowers/plans/2026-08-16-task-18914-library-media-pagination.md

1. Add coherent exact-offset Media DB paging and metadata-only diagnostics.
2. Propagate true offsets and complete active type facets through existing local/scope services.
3. Add exact Media page validation and source-owned requested/applied/retained controller state.
4. Wire dedicated screen authority, generation/unmount fences, restore, selection clearing, focus, and shrink recovery.
5. Render the pager and one bounded complete-type OptionList at both supported geometries.
6. Preserve applied scope and declarative stale-action safety through delete/bulk/undo.
7. Run only touched Media component tests, focused inverses, isolated live proof, reviews, docs, and closeout; do not run the full repository suite per user direction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented authoritative Media paging from the SQLite query through the local
and scope services into a dedicated requested/applied/retained Library
controller. Media now renders exact 20-item pages, a pinned pager, and one
bounded complete-type chooser whose unfiltered value is internal `None`, so
literal stored types such as `All`, `all`, and `ALL` remain selectable.

Generation and lifecycle fences prevent stale page, facet, broad-snapshot, or
unmounted results from replacing the applied page. Page/type changes clear
current-page selection visibly. Delete, bulk delete, Undo, edit, and Trash
restore use validated backing IDs and one durable-write interlock; retained
rows reconcile conservatively, while failed authoritative refreshes leave
unsafe actions disabled and Retry/type recovery available. Media DB diagnostics
now report bounded metadata only on success, reopen, and connection failure.

Evidence:

- Database/service/state/controller owners: 328 passed, including Media DB
  pagination, privacy logging, property, local/scope/off-loop, state, and
  controller tests.
- Mounted UI owners were confirmed in bounded partitions: 104 passed in the
  Media shell owner and 97 passed across the remaining seven direct owners.
  Five pre-existing entry-origin diagnostics were explicitly excluded per user
  direction. Two aggregate-only late load/order failures passed 2/2 in
  isolation; neither bounded partition reproduced them.
- Ruff passed on all 21 changed Python files; the component CSS build and
  generated-bundle parity check passed; `git diff --check` passed.
- Independent final spec and quality/minimality reviews both reported READY
  with no Critical or Important findings.
- Isolated synthetic profiles seeded 65 real Media DB rows and 60 distinct
  types. Real `TldwCli.CSS_PATH`, Media DB, local service, and scope service
  runs passed at 100x30 and 170x48 with 20/20/20/5 pages, a reachable row 20
  and pinned pager, keyboard type commit/cancel, literal `All` separation,
  filtered totals, selection clearing, controlled failure/Retry, mutation
  stale recovery, and detail/Back. Both tmux panes exited 0; privacy sentinels,
  real-profile/foreign DB handles, and TCP listeners were all zero, and the
  real-profile fingerprint was unchanged.
- The live harness exits with `os._exit(0)` only after Textual returned and the
  unmount/evidence writes were flushed, avoiding a third-party interpreter
  finalizer crash observed after an otherwise successful `app.run()`.

Per user direction, repository-wide pytest was not run; only modified/touched Media component and direct-owner gates are claimed.

ADR check: no new ADR was required; this directly implements
`backlog/decisions/067-library-top-level-pagination-contracts.md`. Existing
testing and live-verification lessons covered the incidents encountered, so no
new lesson was added.
<!-- SECTION:NOTES:END -->
