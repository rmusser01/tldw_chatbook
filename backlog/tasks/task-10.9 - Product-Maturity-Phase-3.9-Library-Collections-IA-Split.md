---
id: TASK-10.9
title: 'Product Maturity Phase 3.9: Library Collections IA Split'
status: Done
assignee: []
created_date: '2026-05-08 03:36'
updated_date: '2026-05-08 04:41'
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
- [x] #1 Visible navigation command palette help and destination copy use Watchlists for monitored-source workflows and do not present Collections as part of the top-level Watchlists destination
- [x] #2 Library Collections mode lets users list create select rename and safely delete local collections with updated-at display and honest local-only or sync-unavailable status
- [x] #3 Existing Watchlists active-run Home and Console follow-through remains usable through compatibility route IDs where needed
- [x] #4 Focused automated tests and QA walkthrough evidence prove the split and local Collections management are usable rather than merely renderable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TASK-10.9.1: Watchlists IA split and compatibility labels.
2. TASK-10.9.2: Library Collections display-state and local service contracts.
3. TASK-10.9.3: Library Collections mounted management UI.
4. TASK-10.9.4: QA closeout and tracking.

Primary implementation plan: Docs/superpowers/plans/2026-05-08-phase-3-9-library-collections-ia-split.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed Phase 3.9 with Watchlists split from Collections and Library-owned local Collections management verified. The top-level destination now presents Watchlists only, Collections is discoverable under Library with create select rename delete status and safe delete coverage, Watchlists active-run follow-through remains compatible, QA evidence records focused verification and clean startup smoke, and later-stage risks remain for collection membership server sync Import/Export scoped Study/Search/RAG flows and citation/snippet carry-through.
<!-- SECTION:NOTES:END -->
