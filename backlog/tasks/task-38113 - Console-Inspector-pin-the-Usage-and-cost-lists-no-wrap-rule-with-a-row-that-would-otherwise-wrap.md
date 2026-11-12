---
id: TASK-38113
title: >-
  Console Inspector: pin the Usage and cost list's no-wrap rule with a row that
  would otherwise wrap
status: To Do
assignee: []
created_date: '2026-10-30 05:57'
labels:
  - console
  - inspector
  - tests
  - follow-up
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the TASK-33621.9 review. TASK-33621.9 put text-wrap: nowrap on the Usage and cost list only, and reordered each usage row so basis and cost lead. The painted-frame test passes even with the no-wrap rule removed, because the reordered rows already fit at 80x24 and 235x52. The reorder is what currently satisfies the AC; the no-wrap rule is untested defence in depth, so a later CSS change could drop it unnoticed and bring back a basis label wrapped onto its own line once a row grows (long role names, large token counts, wider cost strings).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A painted-frame test renders a usage row longer than the list width and asserts it stays on one line with the basis label visible
- [ ] #2 The test fails when the no-wrap rule is removed from the Usage and cost list
- [ ] #3 Context and Exchange labels still wrap (the TASK-33621.9 guard stays green)
<!-- AC:END -->
