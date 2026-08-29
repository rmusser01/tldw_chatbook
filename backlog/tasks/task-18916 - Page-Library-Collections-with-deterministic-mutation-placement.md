---
id: TASK-18916
title: Page Library Collections with deterministic mutation placement
status: Done
assignee: []
created_date: '2026-08-15 02:49'
updated_date: '2026-08-29 06:34'
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
- [x] #1 Collection browsing uses the existing exact-total service seam with coherent 20-row pages and deterministic creation-time, case-insensitive-name, stable-ID ordering.
- [x] #2 Create, rename, and restore locate the affected stable ID through one bounded coherent rank-derived owning-page response and select it without walking pages.
- [x] #3 Delete reloads or clamps the current page; external or repeated concurrent shrink follows the approved bounded stale-recovery contract.
- [x] #4 Malformed browse or locator envelopes fail closed, including absent targets, unaligned locations, invalid cardinality, or duplicate identities.
- [x] #5 Collection mutation success remains truthfully committed when follow-up reads fail, with locally reconciled stale state and unsafe actions disabled.
- [x] #6 Collection request generations, unmount fencing, broad-snapshot isolation, focus, restoration, and recoverable error behavior match the approved design.
- [x] #7 Automated SQLite service/state, mounted Textual, geometry, mutation, and isolated live verification pass.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact 20-item Collection pages under the existing ADR-067 contract.
The local service now provides deterministic coherent page and stable-ID locator
reads; strict immutable validators fail closed on malformed page or locator
metadata; a dedicated controller owns generations, one-clamp recovery, retained
stale rows, and Retry; and the retained Collections panel owns the independently
scrolling list plus exact range/page and Previous/Next controls. LibraryScreen
restores only applied page state and uses bounded stable-ID placement for create,
rename, and restore while delete reloads or clamps its current page. Committed
writes remain visibly committed with inert locally reconciled rows when a
follow-up read fails.

Verification: 220 focused tests passed (49 SQLite service, 28 state, 14
controller, 32 mounted Collections, and 97 shared Library entry-lifecycle
tests), plus Ruff and `git diff --check`. Production-shaped 45-row walkthroughs
passed at 100x30 and 170x48 across first, middle, and final pages, including
pager containment, painted copy, selected-row readability, form focus, and edge
focus fallback. An isolated temporary SQLite walkthrough passed create page-1
placement, rename relocation to page 3, delete, restore, injected committed
locator failure, inert stale state, and successful stable-ID Retry without page
walking. The required shared lifecycle gate also exposed and repaired a
pre-existing restored-Skills mount bug at the exact dev base; the incident is
recorded in `backlog/docs/lessons-testing-evidence.md`.

ADR required: yes
ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`
Reason: ADR-067 governs the ordering, locator, freshness, and mutation recovery
contracts implemented by this task; no new ADR was needed.
<!-- SECTION:NOTES:END -->
