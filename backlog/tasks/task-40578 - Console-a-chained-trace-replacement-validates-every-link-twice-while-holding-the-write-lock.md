---
id: TASK-40578
title: >-
  Console: a chained trace replacement validates every link twice while holding
  the write lock
status: In Progress
assignee:
  - '@claude'
created_date: '2026-11-12 23:24'
updated_date: '2026-11-20 16:56'
labels:
  - console
  - trace
  - performance
dependencies:
  - TASK-38298
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chained trace replacement (TASK-38298, ADR-337 amendment 2026-11-06) is written in the IMMEDIATE dispatch-binding transaction. Inside it, the request writer validates the replacement once (every link's range read, plus the check that no live row lies between links), and the replacement writer validates it again before writing. The work is linear in the replaced span. Measured 2026-11-12 on a loaded machine, binding a 1,999-row chain (8 links) took about 73 ms of CPU (109 ms wall) at the median, mostly these reads. At the 256-link cap (about 65,000 rows) the same work may hold the database write lock for seconds, blocking every other writer. Measure it, and consider validating each link once.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Binding a chain is measured at about 2,000 rows and at the 256-link cap, wall and CPU, with the share spent validating links
- [ ] #2 If the duplicate validation is removed or narrowed, every link is still validated once inside the binding transaction, and the tests that pin per-link revalidation stay green
- [x] #3 The measurements and the decision are recorded in the task notes
<!-- AC:END -->

## Identity

Filed 2026-11-12 from the PR #3515 review (finding F4). A sweep of every ref's task paths and every worktree found a maximum of 40546, so this task takes 40578.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first, on unchanged dev code: bind a chain at 1,999 rows (8 links) and at the 256-link cap (65,536 stored rows), warm service, real planner and preparation; repeated medians of wall and thread CPU, the time inside each _validate_replacement_range pass and its lineage reads, load1 with every run.
2. Re-measure after TASK-40577's check lands.
3. Decide from the numbers: keep the duplicate pass if it costs little, or validate once inside the binding transaction (the request writer's pass) and keep the per-link revalidation tests green.
4. Record measurements and decision in the task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Measured first, decided second: the duplicate validation stays.** The write lock is held for about 38 s at the 256-link cap, but validation is 0.6% of that; 99% is the replacement-row lineage trigger, filed as TASK-42130.

**Method.** A scratch probe under `Tests/Chat/` (deleted before commit; kept in the session scratchpad) used the planner tests' real `_Chat` harness and the Compact now shape: record N rows in one send, commit a conversation memory, then send 3 rows, so the whole history is replaced as a chain. A template database was copied with the SQLite backup API before each run. A fresh service prepared the send and then bound it, warm, as the runtime does. `bind_and_mark_dispatch` was timed with `perf_counter` and `thread_time`, after `gc.collect()`. Wrappers timed each `_validate_replacement_range` pass and its `read_lineage_surface_nodes` calls. load1 was read before and after every bind. M4 Pro, shared host.

The cap shape is an upper bound. Its 65,536 rows were recorded live in one send; the setup raised the credential sanitizer's 100,000-node budget to do that, since a captured request otherwise carries at most about 20,000 messages. The validation's lineage reads are at their true maximum regardless: 65,536 rows per pass, live or dead. The trigger's cost follows stored depth, and any 256-link chain spans at least 65,281 stored sequences.

**Baseline: dev d85e90bc9e, code unchanged.** These are medians of unprofiled runs. "Validation" is both passes; "duplicate" is the second pass, in `_replace_surface`.

| Shape | n | load1 | Bind wall | Bind CPU | Pass 1 CPU (lineage reads) | Duplicate pass CPU | Validation share of CPU | Duplicate share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,999 rows, 8 links | 8 | 4.44 | 35.9 ms | 35.6 ms | 3.22 ms (2.80) | 3.21 ms | 18.2% | 9.1% |
| 65,536 rows, 256 links | 2 | 4.9-5.8 | 38.4 s | 38.0 s | 111.7 ms (99.5) | 111.7 ms | 0.59% | 0.29% |

**With TASK-40577's check.**

| Shape | n | load1 | Bind wall | Bind CPU | Pass 1 CPU (lineage reads) | Duplicate pass CPU | Validation share of CPU | Duplicate share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,999 rows, 8 links | 9 | 3.9 | 36.2 ms | 36.2 ms | 3.37 ms (2.96) | 3.31 ms | 18.5% | 9.1% |
| 65,536 rows, 256 links | 3 | 5.2-6.6 | 39.8 s | 38.5 s | 112.0 ms (99.4) | 110.5 ms | 0.58% | 0.29% |

**Attribution.** These runs had TASK-40577's check, with `console_trace_surface_replacements_lineage` dropped from each run's scratch copy of the database:

| Shape | n | load1 | Bind wall | Bind CPU | Validation share | Duplicate share |
| --- | --- | --- | --- | --- | --- | --- |
| 1,999 rows, 8 links | 5 | 3.9 | 13.1 ms | 13.0 ms | 52.8% | 26.1% |
| 65,536 rows, 256 links | 2 | 4.3-4.4 | 294.5 ms | 294.3 ms | 75.9% | 37.9% |

cProfile agrees. At the cap, 37.77 s of a 38.24 s bind was in the 257 `append_surface_replacement` inserts, about 147 ms each. At 2,000 rows the 9 inserts took 24 ms, about 2.7 ms each.

**Decision: keep the replacement writer's second validation pass.**
1. The finding's concern is real: binding at the cap holds the write lock for seconds, about 38 of them. Its cause was not the validation. Removing the duplicate saves 0.11 s of 38.5 s at the cap. At 2,000 rows it saves 3.3 ms of 36 ms, paid once per rewrite: the send after Compact now or Edit & resend.
2. The pass is `_replace_surface`'s own check, in the only writer of replacement rows. It keeps that writer safe if its caller, or the code between the request writer's check and the write, changes later.
3. The lock is the trigger's, so TASK-42130 owns it. Once that cost is gone, validation is most of what remains (76% of a 0.29 s bind at the cap, 38% for the duplicate). Revisit the duplicate with that task's measurements.

AC#2 is conditional and its condition did not arise: no validation was removed or narrowed. The per-link revalidation pins (`test_dispatch_binding_revalidates_every_link`, and TASK-40577's binding test) are green.

No code changed for this task; its commit holds only these notes and the follow-up task file.
<!-- SECTION:NOTES:END -->
