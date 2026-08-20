---
id: TASK-18919
title: Add paged viewing to Library Collection members
status: To Do
assignee: []
created_date: '2026-08-15 02:52'
labels:
  - library
  - pagination
  - collection-members
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
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every member of a large Collection reachable through bounded nested detail pages while preserving Collection context, membership mutations, and source-item navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collection detail exposes coherent exact-total bounded member pages with deterministic stable ordering and complete-scope filtering before slicing.
- [ ] #2 Add, remove, restore, and source-item navigation retain the owning Collection context and place or remove the affected stable member deterministically.
- [ ] #3 Mutation success remains truthful when follow-up reads fail; stale rows and totals are clearly marked and unsafe actions remain disabled until recovery.
- [ ] #4 Loading, empty, failure, Retry, focus, detail/back navigation, and narrow-terminal geometry match the established Library pagination convention.
- [ ] #5 Request generations, unmount fencing, malformed envelopes, concurrent shrink, and late parent-snapshot isolation have regression coverage.
- [ ] #6 Automated service/state and mounted Textual tests plus isolated live verification with a Collection containing more than 40 synthetic members pass.
<!-- AC:END -->
