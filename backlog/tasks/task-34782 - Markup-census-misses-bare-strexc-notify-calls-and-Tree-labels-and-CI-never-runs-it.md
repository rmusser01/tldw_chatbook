---
id: TASK-34782
title: >-
  Markup census misses bare str(exc) notify calls and Tree labels, and CI never
  runs it
status: To Do
assignee: []
created_date: '2026-10-10 18:14'
labels:
  - ci
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-1513 markup-interpolation census only flags interpolation shapes (f-string, %, .format, +). A bare notify(str(exc)) is invisible to it, and about 37 such calls without markup=False exist (for example library_rag_search_controller.py and Evals/snippet_editor.py); exception text can carry brackets and raise MarkupError. Tree labels are not checked at all. derived-artifacts.yml does not run check_markup_interpolation.py, so dev went red in preflight (TASK-34780) without any PR failing. Found by the TASK-34780 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census flags notify calls whose message is runtime text (including bare str(exc)) unless markup=False or an escape is applied
- [ ] #2 Existing bare str(exc) notify sites are fixed or individually justified
- [ ] #3 A required CI job runs check_markup_interpolation.py, so a PR that adds an unsafe site fails
<!-- AC:END -->
