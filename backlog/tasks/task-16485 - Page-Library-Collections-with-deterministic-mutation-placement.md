---
id: TASK-16485
title: Page Library Collections with deterministic mutation placement
status: To Do
assignee: []
created_date: '2026-08-15 02:49'
labels:
  - library
  - pagination
  - collections
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
Make every top-level Collection reachable through 20-item Library pages and keep create, rename, restore, delete, and selection placement deterministic under live data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collection browsing uses the existing exact-total service seam with coherent 20-row pages and deterministic creation-time, case-insensitive-name, stable-ID ordering.
- [ ] #2 Create, rename, and restore locate the affected stable ID through one bounded coherent rank-derived owning-page response and select it without walking pages.
- [ ] #3 Delete reloads or clamps the current page; external or repeated concurrent shrink follows the approved bounded stale-recovery contract.
- [ ] #4 Malformed browse or locator envelopes fail closed, including absent targets, unaligned locations, invalid cardinality, or duplicate identities.
- [ ] #5 Collection mutation success remains truthfully committed when follow-up reads fail, with locally reconciled stale state and unsafe actions disabled.
- [ ] #6 Collection request generations, unmount fencing, broad-snapshot isolation, focus, restoration, and recoverable error behavior match the approved design.
- [ ] #7 Automated SQLite service/state, mounted Textual, geometry, mutation, and isolated live verification pass.
<!-- AC:END -->
