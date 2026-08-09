---
id: TASK-4020
title: Nav ghosting does not prevent mid-word tab cuts after the overflow-menu change
status: To Do
assignee: []
created_date: '2026-08-09 20:30'
labels:
  - navigation
  - regression
  - recritique-2026-08-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09 (RC-02), measured at dev `4d0232358` by the mechanical arm.

task-3200's four-round arc existed to guarantee that a destination tab label is never rendered
mid-word-clipped. At dev tip that guarantee does not hold:

- 80 cols: `⌃6 Watc  More ▾`
- 120 cols: `⌃9 M  More ▾`
- scroll fragments observed: `‹ ts  ⌃5 Roleplay…`, `‹ oleplay…`, `‹ lists…`, `‹ edules…`

No ghosted tabs were observed at all — the bar scrolls with a `‹` indicator instead.

**The ghosting machinery is present** (10 references in `UI/Navigation/main_navigation.py`, 2 in
`css/components/_navigation.tcss`), so this is a failure of effect, not lost code. Leading
hypothesis: dev replaced the in-strip pager with `NavOverflowMenu` while task-3200 was in flight;
the polish batch's rebase (PR #1459) kept the ghosting mechanism and dropped the pager-specific
pieces, but the scroll/paging model the straddle detection was written against changed underneath
it. Re-root-cause against the current overflow model rather than re-patching the old assumptions.

Mitigating: no ghosted tab was clickable-while-invisible (a blank-cell click did not navigate), so
the round-1 interactivity hole stays closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No destination tab label renders mid-word-clipped at 80, 100 or 120 columns, verified by rendered-geometry assertions and live capture
- [ ] #2 The root cause is stated: why the existing ghost/straddle detection stopped producing its effect under the overflow-menu model
- [ ] #3 The scroll-fragment renders (`‹ oleplay…`) are gone or are a deliberate, documented affordance
- [ ] #4 Regression coverage runs against the CURRENT overflow model, and the now-obsolete assumptions in task-3200's tests are corrected rather than left passing vacuously
<!-- AC:END -->
