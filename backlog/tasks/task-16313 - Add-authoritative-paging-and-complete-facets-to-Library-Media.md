---
id: TASK-16313
title: Add authoritative paging and complete facets to Library Media
status: To Do
assignee: []
created_date: '2026-08-15 02:47'
labels:
  - library
  - pagination
  - media
  - privacy
dependencies:
  - TASK-16311
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
