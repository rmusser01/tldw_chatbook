---
id: TASK-10.9
title: 'Product Maturity Phase 3.9: Library Collections IA Split'
status: To Do
assignee: []
created_date: '2026-05-08 03:36'
updated_date: '2026-05-08 03:39'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - phase-3-9-library-collections
dependencies:
  - TASK-10.8
references:
  - Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md
parent_task_id: TASK-10
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split the combined Watchlists+Collections product model by making Watchlists the top-level monitored-source destination and moving local Collections management into Library as a reusable source organization workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Visible navigation command palette help and destination copy use Watchlists for monitored-source workflows and do not present Collections as part of the top-level Watchlists destination
- [ ] #2 Library Collections mode lets users list create select rename and delete local collections with honest local-only or sync-unavailable status
- [ ] #3 Existing Watchlists active-run Home and Console follow-through remains usable through compatibility route IDs where needed
- [ ] #4 Focused automated tests and QA walkthrough evidence prove the split and local Collections management are usable rather than merely renderable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TASK-10.9.1: Watchlists IA split and compatibility labels.
2. TASK-10.9.2: Library Collections display-state and local service contracts.
3. TASK-10.9.3: Library Collections mounted management UI.
4. TASK-10.9.4: QA closeout and tracking.

Primary implementation plan: Docs/superpowers/plans/2026-05-08-phase-3-9-library-collections-ia-split.md.
<!-- SECTION:PLAN:END -->
