---
id: TASK-42130
title: >-
  Console: binding a 256-link trace chain holds the write lock for about 38 s in
  the replacement lineage trigger
status: To Do
assignee: []
created_date: '2026-11-20 17:20'
labels:
  - console
  - trace
  - performance
dependencies:
  - TASK-38298
  - TASK-40578
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A chained trace replacement (TASK-38298, ADR-337 amendment 2026-11-06) writes up to 257 replacement rows in the IMMEDIATE dispatch-binding transaction. Every insert into `console_trace_surface_replacements` fires the `console_trace_surface_replacements_lineage` trigger, whose recursive queries walk the predecessor and replacement node chains toward the segment root, so each row costs time in proportion to the segment's stored depth. A chain therefore costs links x depth.

TASK-40578 measured it on dev d85e90bc9e with TASK-40577's check (M4 Pro, load1 4 to 7, thread CPU of the bind call, warm service, the real planner and preparation):

- 1,999 rows, 8 links: 36.2 ms CPU at the median (n=9); 13.0 ms with the trigger dropped from a scratch copy of the database (n=5), so the trigger is about 23 ms.
- 65,536 stored rows, 256 links: 38.5 s CPU, 39.8 s wall (n=3); 0.29 s with the trigger dropped (n=2). cProfile put 37.8 s of 38.2 s in the 257 inserts, about 147 ms each, against about 2.7 ms each at 2,000 rows.

So a send after Compact now or Edit & resend at the chain cap holds the ChaChaNotes write lock for about 38 seconds. ChaChaNotes connections open with a 15 s busy timeout, so a writer that waits on that lock would give up first (not exercised). A 256-link chain needs at least 65,281 stored sequences in its range whatever its live row count, so the cost at the cap does not depend on how many rows were live. With the trigger cost gone, validating the links would be most of what remains (0.22 s of 0.29 s at the cap, two passes; see TASK-40578's notes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Binding a 256-link chain over 65,536 stored rows is measured before and after the change (wall, CPU, the share spent in replacement-row inserts, load1 with every run)
- [ ] #2 Binding that chain no longer costs time in proportion to the number of links times the segment's stored depth
- [ ] #3 Every replacement row the current lineage trigger rejects is still rejected, and a test pins each refusal
- [ ] #4 Any change to the trigger ships with a schema migration, as CLAUDE.md requires
<!-- AC:END -->

## Identity

Filed 2026-11-20 from TASK-40578's measurements. The highest task id on any ref's task paths and in any worktree was 42112; this task leapfrogs to 42130, which no ref or worktree holds.
