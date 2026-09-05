---
id: TASK-31662
title: >-
  Inspect rail density: single-line rows, right-aligned secondaries, redundancy removal
status: To Do
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, ux, critique-2026-09-05]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P1: ConsoleInspectorSectionRow hard-codes 2 lines per row, so at
80x24 the rail shows 8 pinned lines + a 3-line scroll body whose one
visible row restates the header (net-zero information); 25% of the
Environment section's row-lines are blank; a 2-file diff with two
expansions eats 20 lines at 235x52. Measured rail content width is 30
(80x24) to 36 (200x50) columns, not the 34 the summary budget assumes
(TASK-31629 #12/#13). Redundancies: header summary duplicates the branch
row and counts; Tasks header duplicates its only row in different words.
Owner ruling 2026-09-05: the Local row STAYS as designed (do not cut it).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Rows with empty secondary text occupy one line; rows whose secondary fits beside the primary render on one line (secondary right-aligned)
- [ ] #2 At 80x24 the Environment section at rest shows at least its four top-level rows without scrolling
- [ ] #3 The section header's summary is suppressed (or reduced) while the section is open, so open sections never duplicate their own first rows
- [ ] #4 The Tasks counts row no longer restates the Tasks header verbatim
- [ ] #5 Summary/title budgets derive from the rail's real content width (see TASK-31629 #12/#13) and the widget test pins a width the smallest supported terminal actually produces
<!-- AC:END -->
