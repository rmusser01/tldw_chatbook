---
id: TASK-42130
title: >-
  Console: binding a 256-link trace chain holds the write lock for about 38 s in
  the replacement lineage trigger
status: To Do
assignee: []
created_date: '2026-11-20 17:20'
updated_date: '2026-11-20 17:55'
labels:
  - console
  - trace
  - performance
dependencies:
  - TASK-38298
  - TASK-40578
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A chained trace replacement (TASK-38298, ADR-337 amendment 2026-11-06) writes up to 257 replacement rows in the IMMEDIATE dispatch-binding transaction. Every insert into `console_trace_surface_replacements` fires the `console_trace_surface_replacements_lineage` trigger, whose recursive queries walk the predecessor and replacement node chains toward the segment root, so each row costs time in proportion to the segment's stored depth. A chain therefore costs links x depth.

TASK-40578 measured it on dev d85e90bc9e with TASK-40577's check (M4 Pro, load1 4 to 7, thread CPU of the bind call, warm service, the real planner and preparation):

- 1,999 rows, 8 links: 36.2 ms CPU at the median (n=9); 13.0 ms with the trigger dropped from a scratch copy of the database (n=5), so the trigger is about 23 ms.
- 65,536 stored rows, 256 links: 38.5 s CPU, 39.8 s wall (n=3); 0.29 s with the trigger dropped (n=2). cProfile put 37.8 s of 38.2 s in the 257 inserts, about 147 ms each, against about 2.7 ms each at 2,000 rows.

So a send after Compact now or Edit & resend at the chain cap holds the ChaChaNotes write lock for about 38 seconds. ChaChaNotes connections open with a 15 s busy timeout, so a writer that waits on that lock would give up first (not exercised). A 256-link chain needs at least 65,281 stored sequences in its range whatever its live row count, so the cost at the cap does not depend on how many rows were live. With the trigger cost gone, validating the links would be most of what remains (0.22 s of 0.29 s at the cap, two passes; see TASK-40578's notes).

**Realistic sizes.** The TASK-40577/40578 review measured bind CPU on the branch head in the Compact now shape:

| Rows | Links | Bind CPU | Bind wall | load1 |
| --- | --- | --- | --- | --- |
| 2,000 | 8 | 36 ms | 36 ms | 3.5 |
| 5,000 | 20 | 164 ms | 164 ms | 3.4 |
| 10,000 | 40 | 1.03 s | 7.0 s | 82 |

The cost grows about as links x stored depth. A captured request carries at most about 20,000 messages, which is about 80 links. That means several seconds of write lock on CPU alone, and several times that in wall time under load. The trigger's per-row cost also grows with depth for an ordinary single replacement: about 147 ms per row at a depth of 65,536.

**What the user hits.** After Compact now or Edit & resend in a long chat, the next send stalls before dispatch while binding holds the write lock. Other ChaChaNotes writers wait on that lock, and they can fail with "database is locked" once the 15 s busy timeout runs out.

`Tests/Benchmarks/test_console_trace_chain_binding.py` reruns these shapes for AC#1. It is opt-in, set with `TLDW_TRACE_CHAIN_BINDING_BENCH=1`; `TLDW_TRACE_CHAIN_BINDING_ROWS` sets the sizes, and `TLDW_TRACE_CHAIN_BINDING_DROP_TRIGGER=1` measures with the trigger dropped, for attribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Binding is measured before and after the change at about 20,000 rows (about 80 links, the most a captured request reaches) and at the 256-link cap (65,536 stored rows): wall, CPU, the share spent in replacement-row inserts, and load1 with every run
- [ ] #2 Binding a chain no longer costs time in proportion to the number of links times the segment's stored depth
- [ ] #3 Every replacement row the current lineage trigger rejects is still rejected, and a test pins each refusal
- [ ] #4 Any change to the trigger ships with a schema migration, as CLAUDE.md requires
<!-- AC:END -->

## Identity

Filed 2026-11-20 from TASK-40578's measurements. The highest task id on any ref's task paths and in any worktree was 42112; this task leapfrogs to 42130, which no ref or worktree holds.
