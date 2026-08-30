---
id: TASK-24455
title: Screen switches perform 598 query_one lookups
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single screen switch performs 598 `query_one` calls. The hot sites are
`Widgets/Console/console_bounded_section.py::_reconcile` (61),
`UI/Console_Modules/left_rail.py::_mounted_descriptors` (56),
`UI/Console_Modules/left_rail.py::_run_allocation_reconcile` (4 call sites x 28 each), and
`Widgets/destination_rail.py::sync_open` (36).

Each `query_one` walks the DOM. These are reconcile paths that re-resolve the same handles
repeatedly within a single pass rather than resolving once and reusing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `query_one` calls per screen switch are reduced by at least half against the pre-change baseline
- [ ] #2 Repeated lookups of the same node within one reconcile pass resolve once and are reused
- [ ] #3 Rail allocation, bounded-section reconcile and destination-rail open state behave identically after the change
- [ ] #4 A guard pins the per-switch DOM query count so it cannot silently regress
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
PARTIALLY ADDRESSED as a side effect; not directly worked.

`query_one` per screen switch measured 598 before this pass and 482 after, purely from the
composer guards in task-24453 -- the composer participates in screen switches too. The sites this
task names were NOT touched: `console_bounded_section._reconcile` (61 per switch),
`left_rail._mounted_descriptors` (56), `left_rail._run_allocation_reconcile` (4 call sites x 28),
`destination_rail.sync_open` (36). Each re-resolves the same handles repeatedly within one
reconcile pass.

The AC (at least half) is not met: 598 -> 482 is -19%.
<!-- SECTION:NOTES:END -->
