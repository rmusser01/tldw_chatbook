---
id: TASK-31663
title: >-
  Inspect rail focus visibility and keyboard reachability
status: To Do
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, a11y, critique-2026-09-05]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P1, SGR-measured: all five focus-indicator styles in the rail
measure 1.03-1.79:1 against their unfocused state (below the 3:1 non-text
floor); five button stops show no plain-text change at all; one Tab stop
(stop 3 / summary block) has NO indication in either capture at both
sizes; Tab from the composer never reaches the rail (40 presses — a
hidden-but-focusable left-rail widget breaks the route); the section inner
scrollbar thumb renders fg==bg (1.00:1). Related: TASK-31624 (n/p ring)
now carries the trapdoor evidence. Prior art: TASK-24702 found a pure
background tint CANNOT clear 3:1 on this theme — the mechanism must be
shape/outline, not a stronger tint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every focusable rail widget has a visible focus indication in a plain-text capture (shape/glyph carrier, not tint-only), including buttons and chevrons
- [ ] #2 No Tab stop in the rail is indication-free; the stop-3 gap is fixed or that widget is removed from the focus order
- [ ] #3 The hidden-but-focusable left-rail widget is fixed so Tab routing from the composer is not silently absorbed (or the root cause is filed against the left rail with evidence)
- [ ] #4 The section scrollbar thumb is visible against its track at both supported sizes
<!-- AC:END -->
