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
updated_date: 2026-09-28 23:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DB/base_db.py (~738) installs set_trace_callback on every ChaChaNotes connection only to notice BEGIN/COMMIT/ROLLBACK. CPython then renders the expanded SQL, hex-encoding every BLOB parameter, for the statement and for every trigger and FTS sub-step. Inserting a message with a 3 MiB image measured 1,203 ms versus 9 ms for text, and the Console durable-turn commit holds BEGIN IMMEDIATE for that time. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-04; every issue with file:line is listed under PERF-04 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transaction-boundary detection no longer requires a trace callback that expands bound parameters
- [x] #2 The semantic-mutation guard keeps its fail-closed behaviour (existing guard tests pass)
- [x] #3 Inserting a message with a 3 MiB image through CharactersRAGDB takes under 50 ms in a pinned test or benchmark
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
- register_semantic_mutation_guard uses the listener on quiescent connections (all ChaChaNotes connections; the factory is enforced at open) and keeps trace_transaction as a fallback for any other connection type.
- Since these connections run isolation_level=None, every transaction boundary is observable this way.

Fail-closed behaviour is preserved and now pinned. Inside an authorization scope, COMMIT and BEGIN served from the statement cache bypass the authorizer's prepare-time denial. The generation check still refuses the guarded write, and _assert_current_transaction still raises semantic_mutation_transaction_changed.

Measured (isolated profile, load avg ~30):
- 3 MiB image add_message: 1,217 ms -> 28.8 ms median
- text add_message: 10.3 -> 6.6 ms

Tests: Tests/DB/test_semantic_guard_transaction_boundaries.py adds 4 tests. The no-trace test failed first (36 traced statements per add_message); the 3 preservation tests pass before and after.

Regression: Tests/DB, Tests/ChaChaNotesDB and the semantic-guard / trace-collector Chat and UI tests fail the identical 294 pre-existing ids on branch and base (TASK-33370/33371), 0 new failures, and the branch has 4 more passes.

Ruff findings on base_db.py are unchanged. Preflight passes.

Files: tldw_chatbook/DB/base_db.py, Tests/DB/test_semantic_guard_transaction_boundaries.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
