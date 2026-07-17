---
id: TASK-255
title: Remove or wire the orphan research route
status: To Do
assignee: []
created_date: '2026-07-12 14:12'
labels:
  - cleanup
  - ui
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The research route is registered in UI/Navigation/screen_registry.py:90 and resolves to ResearchScreen, but no shell destination, legacy-route mapping or navigation call ever targets it. A registered screen with no entry point is dead weight and confuses route audits. Either give it a real navigation entry or remove the registration. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The research route is reachable via shell navigation or its registration is removed
- [ ] #2 Route inventory and screen registry are consistent with each other
<!-- AC:END -->
