---
id: TASK-16483
title: Add authoritative paging and complete facets to Library Media
status: In Progress
assignee: []
created_date: '2026-08-15 02:47'
updated_date: '2026-08-16 15:14'
labels:
  - library
  - pagination
  - media
  - privacy
dependencies:
  - TASK-16481
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
- [ ] #1 Media filtering and type selection apply to the complete source before coherent exact-total database-level 20-row offset paging.
- [ ] #2 Deep-page reads are bounded and deterministic, with stable Media ID tie-breaking and exact validated envelope cardinality and identities.
- [ ] #3 The Media type chooser exposes the complete distinct type set in one bounded keyboard-accessible OptionList-style control with active and cancel behavior.
- [ ] #4 Media current-page selection clears with visible notice on page, filter, sort, or type scope change, and stale mutation recovery disables unsafe actions.
- [ ] #5 Touched Media read diagnostics contain metadata only and never query text, titles, bodies, stable private IDs, filesystem paths, or credentials on success or error.
- [ ] #6 Media request generations, unmount fencing, broad-snapshot isolation, focus, restoration, concurrent shrink, and recoverable error behavior match the approved design.
- [ ] #7 Automated database/service/state, mounted Textual, geometry, privacy, mutation, and isolated live verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: Direct implementation of ADR-067's approved Media paging, facet, stale-recovery, and privacy contracts.

Detailed plan: Docs/superpowers/plans/2026-08-16-task-16483-library-media-pagination.md

1. Add coherent exact-offset Media DB paging and metadata-only diagnostics.
2. Propagate true offsets and complete active type facets through existing local/scope services.
3. Add exact Media page validation and source-owned requested/applied/retained controller state.
4. Wire dedicated screen authority, generation/unmount fences, restore, selection clearing, focus, and shrink recovery.
5. Render the pager and one bounded complete-type OptionList at both supported geometries.
6. Preserve applied scope and declarative stale-action safety through delete/bulk/undo.
7. Run only touched Media component tests, focused inverses, isolated live proof, reviews, docs, and closeout; do not run the full repository suite per user direction.
<!-- SECTION:PLAN:END -->
