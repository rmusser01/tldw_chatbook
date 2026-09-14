---
id: TASK-18601
title: >-
  Agent run step log is a single JSON blob column and does not scale to the
  raised step budget
status: Done
assignee:
  - '@codex'
created_date: '2026-08-18 20:30'
updated_date: '2026-09-12 17:43'
labels:
  - agents
  - database
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentRunsDB.append_steps` stores a run's entire step log as one JSON string in
the `agent_runs.steps` column: it SELECTs the existing blob, `json.loads` it,
extends the list, and `json.dumps` the whole thing back. `AgentService._persist`
calls it once at end of run, so the write itself is O(n), not O(n^2).

That design was sized for the Console's old 96-step budget. TASK-18600 raised
the shipped step budget to 25000 (owner decision: allow long-running, expensive
sessions). Each step's `result` is capped at 2000 characters by
`agent_runtime.run_agent_loop`, so a worst-case run can now serialize a
tens-of-megabytes JSON blob into one column -- and re-parse all of it every time
the run log is opened, on the UI thread.

Nothing observed failing yet: this is a scaling limit reached by a deliberate
config change, filed at the time the ceiling was raised rather than after a user
hits it. Realistic runs are far smaller (most steps are a few hundred bytes),
which is why TASK-18600 shipped the number as specified instead of lowering it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run with 25000 recorded steps persists and re-opens without a user-visible stall in the run-log viewer.
- [x] #2 Reading a run's metadata (status, budget, result) does not require parsing its full step log.
- [x] #3 The run-log viewer can render a long run without holding every step in memory at once.
- [x] #4 Existing runs stored in the current blob format remain readable after the change.
- [x] #5 When an expanded rail checks a run before its first complete log record exists, the log action becomes available after a later append through bounded off-thread retries; a still-admitted probe is not restarted, cancelled generations cannot publish stale results, and collapsed steady-state ticks perform no log I/O.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/082-console-per-chat-private-scratch-space.md
Reason: bounded presentation/read-path implementation of the existing byte-framed segment log and scratch-read lease contracts.

1. Add a bounded byte-framed page reader with oversized-record continuation and exact child filtering.
2. Expose each page through the current bridge authority and replace all-record availability scans.
3. Add worker-loaded Previous, Next and First navigation retaining one page and bounded cursors.
4. Verify large UTF-8 records, sparse segments, incomplete/corrupt input, revoked authority and mounted lifecycle.
5. Run targeted checks and independent task/combined review; close only remaining viewer AC.
Spec: Docs/superpowers/specs/2026-09-12-bounded-console-run-log-design.md
Plan: Docs/superpowers/plans/2026-09-12-bounded-console-run-log.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Historical database part A evidence (2026-08-21), preserved below. These are historical measurements, not fresh paging verification. The remaining viewer now reads lossless filesystem segments; its new bounded-page work follows the current plan.

### Measurement (2026-08-21) — the premise, quantified

`append_steps` (`DB/AgentRuns_DB.py`) reads the ENTIRE step log, JSON-parses
it, extends the list, re-serializes it and rewrites the whole column -- on
every append. That is O(n) per step and O(n^2) per run.

Measured against a real `AgentRunsDB`, one ~200-byte step per append:

    append #    1:    0.05 ms
    append #  100:    0.13 ms
    append #  500:    0.49 ms
    append # 1000:    0.98 ms
    append # 1500:    1.42 ms
    append # 2000:    2.18 ms
    2,000 appends total: 2.09s

Per-append cost grows linearly with log size (44x from the 1st to the
2,000th), confirming the quadratic total. Extrapolating the same curve to the
25,000-step budget AC #1 names gives **~5.4 minutes** of pure database churn
for one run -- and that is write cost alone, before the viewer reads it back.

**Scope.** This is an arc, not a single change: a child `agent_run_steps`
table plus a schema bump (AgentRuns_DB is at v4, and its migration mechanism
is `CREATE TABLE IF NOT EXISTS` on every open), a compatibility read path so
existing blob-format runs stay readable (AC #4), and a viewer change to page
steps rather than hold them all in memory (AC #3). Suggested split:

* **A** -- child table + dual-read (new writes go to rows, reads prefer rows
  and fall back to the blob). Closes AC #1, #2, #4 and is independently
  shippable.
* **B** -- viewer paging over the new table. Closes AC #3.
* **C** -- optional backfill of historical blobs, once A has soaked.

Not started here; the measurement is recorded so the arc can be planned
against a number rather than an adjective.

### Part A shipped (2026-08-21) — AC #3 (viewer paging) still open

Steps moved out of the rewritten `agent_runs.steps` blob into a child table;
`append_steps` is now an INSERT instead of a read-modify-write of the whole
log.

**Measured, independently of the implementer's own numbers:**

    ms per append   #1      #500    #2000
    before          0.05    0.49    2.18     (44x growth -- quadratic)
    after           0.088   0.024   0.024    (flat)

At the 25,000-step budget AC #1 names that is roughly **5.4 minutes of write
churn reduced to under a second**.

**Compatibility (AC #4) verified by probe, not by assertion.** A run whose
steps live only in the old blob reads back exactly (5/5 steps, identical
content). A MIXED run -- legacy blob plus new appends -- returns all of them
with blob steps first and new steps last, so a legacy run appended to after
the change is not reordered or truncated.

**Still open — AC #3**, the run-log viewer paging so a long run is not held
in memory at once. Unchanged here; that is part B of the split recorded above.
Also left deliberately: `list_runs`/`undelivered_wake_runs` still take the
full-hydration path (disproportionate for part A, documented in the code).

**Incidental find:** `Tests/Chat/test_console_agent_swap.py::_all_runs`
bypassed the DB API and read `agent_runs` with raw SQL, so it under-reported
steps once they moved. A pre-existing test-only bug -- fixed here, but worth
noting as a pattern: a test that reaches around its own API stops testing the
API and starts pinning the storage layout.

### Part B completed (2026-09-12)

The Console viewer now pages the actual lossless filesystem segments, preserving primary-tree and exact-child filtering. Each page contains at most 100 fragments and 256,000 content bytes; large UTF-8 records continue without truncation. The modal retains one current page and at most 256 previous cursors. First/Previous/Next run through workers; errors retain the last page and Close/Escape remain usable during reads. Empty advancing scan pages remain navigable.

Target and parent ownership lookups use metadata-only reads on workers. Every page reacquires the owning scratch authority; neither a persisted ID nor a cursor recreates access. UI selection checks use existing in-memory turn/run identity and the attached runtime, without lazy initialization or database work. Availability retries negatives after one second on existing expanded ticks, never restarts a still-admitted probe, and settles stale/cancelled generations safely. Collapsed steady-state ticks perform no log I/O. No timer, migration, dependency or storage-format change was added. Existing ADR-082 applies.

Verification: reader/codec/search/bridge selection passed 119 tests, including exact reconstruction of a 3,000,000-byte UTF-8 record, sparse segments and instrumented read budgets. The final integration-fix selection passed 61 tests (30 deselected) across Tests/Chat/test_console_agent_tool_result_cap.py and the log-related Tests/UI/test_console_agent_rail.py / test_console_run_log_paging.py nodes. It forbids full step hydration and covers primary/child later pages, queued/running cancellation, stale-target return, slow metadata lookup, absent-runtime behavior, history eviction and native-cell layout at 80x24/160x48. The earlier log-modal/dismissal scope passed 8 tests and CSS/token guards passed 6. These selections overlap and are not an aggregate unique test count. Scoped Ruff/format/whitespace checks pass without added diagnostics; the reviewed production diagnostic inventory reports no drift.

Implementation commits: dbd04f9e5e, a8c4c1972e, 2174c927ac and a70687e176. Both independent task reviews and their scoped fix re-reviews are clean. The combined reader-to-bridge-to-modal review checked bounded framing, metadata-only ownership, per-page leases and cursor-only retention. User guide and design/plan document final behavior. Original Part A measurements and checked criteria above remain historical evidence, not a newly repeated database benchmark.

Limits: load_run_log_text remains an explicitly unbounded compatibility API for a concrete non-viewer skill redaction consumer; production viewing uses pages. Verification used isolated real SQLite/files and mounted Textual with deterministic providers, not a live provider or full repository sweep. Three pre-existing Tests/UI/test_console_modal_dismissal.py inventory assertions remain: test_console_modal_launch_declarations_match_runtime_construction, test_console_modal_inventory_matches_runtime_ast_and_transitive_launches, and test_task2_modal_contract_table_is_complete_and_adopted. Their exact failures are unchanged with original product-source controls and branch-base declaration/table files; actual log-modal dismissal passes. Existing RequestsDependencyWarning and foreign pytest garbage-directory cleanup warnings remain visible.
<!-- SECTION:NOTES:END -->
