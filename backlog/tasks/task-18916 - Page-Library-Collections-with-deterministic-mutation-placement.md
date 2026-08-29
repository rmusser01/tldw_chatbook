---
id: TASK-18916
title: Page Library Collections with deterministic mutation placement
status: In Progress
assignee: []
created_date: '2026-08-15 02:49'
updated_date: '2026-08-29 05:19'
labels:
  - library
  - pagination
  - collections
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Repair the mounted Collections baseline and isolate startup probes.
2. Add strict immutable 20-row page and stable-ID locator contracts.
3. Add deterministic SQLite ordering and one-transaction owning-page locator reads.
4. Add a source-owned Collections browse controller with generations, one-clamp recovery, and stale reconciliation.
5. Render the pager inside the retained Collection list pane with bounded geometry and safe action gates.
6. Wire page restoration, focus, create/rename/restore locator placement, and delete recovery in LibraryScreen.
7. Verify targeted automation, 100x30/170x48 geometry, isolated live mutations, docs, and Backlog hygiene.

ADR required: yes
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: ADR-067 already governs deterministic top-level ordering, stable-ID owning-page reads, freshness, and mutation recovery.
<!-- SECTION:PLAN:END -->
