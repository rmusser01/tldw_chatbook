---
id: TASK-32659
title: Complete remaining Collections browse controls
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-15 22:00'
labels:
  - library
  - collections
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up the Collections reader review with the remaining control-path findings: Clear preserves the active text search, More saved searches has no dispatch handler, and repeated Archive can replace the original Undo receipt. Confirm each through its user journey before repairing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clear removes the search and form filters responsible for the empty-result state, with valid sorting and truthful results.
- [ ] #2 More saved searches reaches additional real saved searches and maintains active authority and scope.
- [ ] #3 Repeated Archive cannot overwrite the original reversible status or present an action that silently changes its Undo meaning.
- [ ] #4 Targeted and native evidence qualify the repaired controls at wide and compact sizes.
<!-- AC:END -->
