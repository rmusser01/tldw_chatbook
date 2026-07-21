---
id: TASK-420
title: >-
  Skills - remove dead chrome (inert Model override field, mis-scoped u
  shortcut)
status: To Do
assignee: []
created_date: '2026-07-21 15:19'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. Two no-op affordances on the Skills surface: (1) the Model override input is live, editable and dirty-marking but annotated 'Not applied in v1.' - it does nothing and can even trigger unsaved-changes friction; (2) the Library footer advertises 'u - use Library context in Console' on the Skills canvas but the action early-returns unless the Search/RAG row is selected. NNG heuristic 8 (aesthetic and minimalist design).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Model override is hidden or read-only-disabled until it has effect and no longer marks the editor dirty,The footer u shortcut hint appears only on rows where the action works,Covered by tests
<!-- AC:END -->
