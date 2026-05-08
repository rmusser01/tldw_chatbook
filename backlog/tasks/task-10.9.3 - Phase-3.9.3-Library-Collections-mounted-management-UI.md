---
id: TASK-10.9.3
title: 'Phase 3.9.3: Library Collections mounted management UI'
status: To Do
assignee: []
created_date: '2026-05-08 03:37'
labels:
  - product-maturity
  - phase-3-9-library-collections
  - library
  - ui
dependencies:
  - TASK-10.9.2
references:
  - Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md
parent_task_id: TASK-10.9
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mount local Collections management inside the existing Library shell so users can create select rename and delete collections from Library.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library Collections mode renders list detail and inspector regions inside the existing Library shell
- [ ] #2 Users can create select rename and delete local collections with persistent visible status and recovery copy
- [ ] #3 The UI does not expose server sync or collection-scoped RAG Study or Console actions as enabled unless implemented
- [ ] #4 Mounted tests cover the management workflow and preserve Gate 1.6 Library Search/RAG regressions
<!-- AC:END -->
