---
id: TASK-31502
title: Quiescence registry taxes every ChaChaNotes statement with a shared lock
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - db
dependencies: []
priority: medium
---

## Description (the why)

`DB/base_db.py:35-410` (`SQLiteConnectionQuiescenceRegistry`,
`_QuiescentSQLiteCursor`) wraps the ChaChaNotes connection
(`ChaChaNotes_DB.py:3339`, `:22895`) so every single
`execute`/`executemany`/`executescript` acquires and releases a process-wide
`Condition(RLock())` and calls `notify_all()`, with a second pair per
`transaction()` block and a third around connection acquisition. It exists so
trace-compaction VACUUM can quiesce connections -- a rare maintenance event --
but the tax is unconditional. Measured (2026-09-04 review, file-backed DB,
20k statements, reproduced twice per arm): per-statement 0.82 -> 1.94 us
(+137%), per-transaction block 18.2 -> 23.3 us (+29%); raw sqlite floor
0.51 us. Small in absolute terms but on the hottest layer, growing with
cross-thread contention, and multiplied by the TASK-31501 1 Hz loop.
Evidence: `Docs/Design/2026-09-04-holistic-perf-review.md` section 3.

## Acceptance Criteria (the what)

- [ ] The uncontended path (no quiesce requested or in progress) adds no lock acquisition and no `notify_all()` per statement (e.g. a relaxed pending-flag check before touching the condition variable)
- [ ] Quiesce correctness is preserved: a quiesce request still drains in-flight statements and blocks new ones (existing quiescence tests stay green, including under concurrent threads)
- [ ] The micro-benchmark shows per-statement overhead through `get_connection()` within ~20% of the pre-registry baseline (~0.8-1.0 us on the review machine)
