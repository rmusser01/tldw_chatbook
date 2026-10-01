---
id: TASK-33263
title: 'PERF-04: ChaChaNotes trace callback makes SQLite hex-expand every BLOB parameter'
status: Done
created_date: 2026-09-28 18:02
labels:
- performance
- database
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
assignee:
- '@claude'
updated_date: 2026-10-01 07:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DB/base_db.py (~738) installs set_trace_callback on every ChaChaNotes connection only to notice BEGIN/COMMIT/ROLLBACK. CPython then renders the expanded SQL, hex-encoding every BLOB parameter, for the statement and for every trigger and FTS sub-step. Inserting a message with a 3 MiB image measured 1,203 ms versus 9 ms for text, and the Console durable-turn commit holds BEGIN IMMEDIATE for that time. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-04; every issue with file:line is listed under PERF-04 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transaction-boundary detection no longer requires a trace callback that expands bound parameters
- [x] #2 The semantic-mutation guard keeps its fail-closed behaviour (existing guard tests pass)
- [x] #3 Inserting a message with a 3 MiB image through CharactersRAGDB measured under 50 ms (28.8 ms median, isolated profile), and a pinned test fails if the hex-expansion cost returns (fastest of five inserts above 250 ms)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: a test proving managed ChaChaNotes connections route no statements through the trace callback (it counted 36 per add_message), plus preservation tests for the fail-closed guarantee:
   - cached COMMIT/BEGIN inside an authorization scope refuses the guarded write;
   - an authorized update in its own transaction still succeeds;
   - a C-level commit is noticed before the next statement.
2. GREEN: _QuiescentSQLiteConnection reports transaction boundaries. Its cursor compares in_transaction before and after every execute/executemany/executescript (executescript always reports one), and commit()/rollback() report too. register_semantic_mutation_guard uses that listener on quiescent connections and keeps the trace callback only as a fallback for other connection types.
3. Benchmark a 3 MiB image add_message, base vs branch, in an isolated profile.
4. Regression: Tests/DB, Tests/ChaChaNotesDB and the semantic-guard / trace-collector Chat tests, branch vs base failing-id sets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The semantic mutation guard no longer uses a trace callback on managed ChaChaNotes connections.

Why: on Python 3.12, set_trace_callback receives the EXPANDED SQL, so SQLite rendered every bound BLOB as hex for each statement and each trigger/FTS step, only so the guard could spot BEGIN/COMMIT/ROLLBACK.

How:
- _QuiescentSQLiteConnection now reports transaction boundaries through set_transaction_boundary_listener.
- Its cursor compares in_transaction before and after every execute/executemany/executescript. The check before a statement also catches a C-level commit from 'with conn:'. executescript always reports a boundary.
- commit()/rollback() report as well.
- A statement whose first keyword is BEGIN, COMMIT or ROLLBACK is always a boundary -- the trace callback's own rule. `ROLLBACK TO <savepoint>` keeps `in_transaction` True yet ends the guarded work; review on #2894 caught the listener missing it. RELEASE was never a boundary and still is not. The keyword boundary is reported only when the statement ran: a COMMIT the authorizer refuses never executes, and reporting it advanced the generation and refused the scope's next legitimate write (review on #2894). A failed call still reports a real in_transaction change, and a script reports a boundary even when it fails.
- Every cursor the connection creates is tracked (review on #2894): a caller's cursor type is combined with the tracked cursor through a cached subclass that puts the tracked cursor first, so its statement methods bracket the caller's overrides even without super(); plain sqlite3.Cursor maps to the tracked cursor. A factory that cannot be made tracked fails closed with TypeError: a non-Cursor factory, or a tracked subclass that replaces execute/executemany/executescript (the tracked cursor cannot precede its own subclass in the MRO).
- A stale cursor's failed call (its connection already closed) releases its quiescence use token, so it cannot pin the registry and block maintenance.
- register_semantic_mutation_guard uses the listener on quiescent connections (all ChaChaNotes connections; the factory is enforced at open) and keeps trace_transaction as a fallback for any other connection type.
- Since these connections run isolation_level=None, every transaction boundary is observable this way.

Fail-closed behaviour is preserved and now pinned. Inside an authorization scope, COMMIT and BEGIN served from the statement cache bypass the authorizer's prepare-time denial. The generation check still refuses the guarded write, and _assert_current_transaction still raises semantic_mutation_transaction_changed.

Measured (isolated profile, load avg ~30):
- 3 MiB image add_message: 1,217 ms -> 28.8 ms median
- Pinned (review on #2894): test_a_3_mib_image_message_inserts_without_the_hex_expansion_cost asserts the fastest of five inserts stays under 250 ms. A median read 313 ms with two 8-worker suites running, so the pin uses the fastest sample: load inflates some samples, the hex-rendering cost inflates every one. With the trace callback restored the fastest of five takes 960 ms. AC#3 was reworded from "under 50 ms in a pinned test" to match: 50 ms is the measured result, not a CI-stable bound.
- text add_message: 10.3 -> 6.6 ms

Tests, all in Tests/DB/test_semantic_guard_transaction_boundaries.py:
- managed connections never trace statements (failed first: 36 traced statements per add_message);
- fail-closed preservation: cached COMMIT+BEGIN inside a scope is refused, an authorized update in its own transaction still succeeds, a C-level commit is seen before the next statement (these pass before and after);
- scripts: one that ends and restarts a transaction reports a boundary, and advances the managed guard generation;
- direct commit()/rollback() each notify the listener and advance the managed generation;
- cursor factories: any factory (plain Cursor, an observing subclass, one overriding every statement method without super()) still reports the C-level-commit-then-BEGIN boundary; the composition rule unit test (tracked cursor first, cached, plain Cursor and tracked types pass through); a non-class factory and a tracked subclass replacing a statement method are refused;
- a cursor used after its connection closed releases its use token.
The review-driven tests each failed before their fix (generation 9 == 9 for the factory cases) or under a negative control.

Regression: Tests/DB, Tests/ChaChaNotesDB and the semantic-guard / trace-collector Chat and UI tests fail the identical 294 pre-existing ids on branch and base (TASK-33370/33371), 0 new failures, and the branch has 4 more passes. Re-run 2026-09-30 after rebasing onto dev dfee4bf4c6: Tests/DB 230 failed vs dev's 232, no new failures (the two apparent ones, the v48 SIGKILL migration test and an FTS cache test, fail on dev too).

Ruff findings on base_db.py are unchanged. Preflight passes.

Files: tldw_chatbook/DB/base_db.py, Tests/DB/test_semantic_guard_transaction_boundaries.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
