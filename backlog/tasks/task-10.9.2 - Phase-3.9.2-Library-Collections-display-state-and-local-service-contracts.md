---
id: TASK-10.9.2
title: 'Phase 3.9.2: Library Collections display-state and local service contracts'
status: To Do
assignee: []
created_date: '2026-05-08 03:36'
labels:
  - product-maturity
  - phase-3-9-library-collections
  - library
dependencies:
  - TASK-10.9.1
references:
  - Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md
parent_task_id: TASK-10.9
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Library-owned pure state and local service boundaries for named Collections without depending on Watchlist or read-it-later services.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pure Collection summary detail action and panel state models cover ready empty loading error invalid-name updated-at and local-only sync states
- [ ] #2 A LibraryCollectionsService contract supports list get create rename and delete operations with validation and stable IDs
- [ ] #3 The first local adapter persists collections in a repo-appropriate versioned local store with foreign-key safety and future-safe membership uniqueness
- [ ] #4 Focused pure tests cover normal empty invalid and failure states without mounting Textual widgets
<!-- AC:END -->
