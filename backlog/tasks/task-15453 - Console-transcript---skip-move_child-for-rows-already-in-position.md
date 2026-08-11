---
id: TASK-15453
title: Console transcript: skip move_child for rows already in position
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified first-hand: `_reconcile_rows` (`Widgets/Console/console_transcript.py:2314-2318`) calls `move_child` for every already-mounted row on every pass, unconditionally. Each `move_child` performs several O(rows) NodeList scans plus `refresh(layout=True)` plus a DOM-version bump that invalidates arrangement/query caches — even when the row is already in place. At ~2 rows per message, a 500-message conversation is ~1,000 rows, and the pass repeats on every 0.2 s streaming tick and on every transcript click (selection triggers a full reconcile). This predates the July task-259 work (a blind spot, not a regression — the content-signature diffing is intact and load-bearing).

Fix direction: track the expected index and skip the move when the widget is already in position; real order changes only occur via prune/variant/branch operations. Stability constraint: the reconciler carries subtle lifecycle guards (the closing/pruning abandon paths and the phantom-mount backstop at `:2306-2313`) — preserve them, and pin ordering behavior with tests covering prune, variant swap, and branch navigation before optimizing. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A steady-state reconcile pass (no order change) issues zero move_child calls (evidence)
- [ ] #2 Ordering still correct after prune, variant swap, and branch navigation (tests)
- [ ] #3 Reconcile pass time on a 500+-message transcript measured before/after and recorded
<!-- AC:END -->
