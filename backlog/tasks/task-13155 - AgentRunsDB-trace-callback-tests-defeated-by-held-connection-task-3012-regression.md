---
id: TASK-13155
title: >-
  AgentRunsDB trace-callback tests defeated by held connection (task-3012
  regression)
status: Done
assignee: []
created_date: '2026-08-09 16:47'
updated_date: '2026-08-09 19:20'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three tests in Tests/DB/test_agent_runs_db.py fail on the current dev baseline, unrelated to any change in this branch. Confirmed as pre-existing during the supervisor-fleet PR-1 program (Task 7 battery + earlier task ledger notes): AgentRunsDB holds a connection open in a way that defeats the trace-callback spies these tests rely on (the same held-connection shape flagged by task-3012's fix elsewhere).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_count_subagents_by_conversation_batches_single_query passes against a real, non-held connection
- [ ] #2 test_transaction_begins_immediate_not_deferred passes, confirming BEGIN IMMEDIATE semantics are actually observed by the trace callback
- [ ] #3 test_count_runs_does_not_materialize_rows_beyond_a_single_count passes
- [ ] #4 Root cause documented: which AgentRunsDB code path holds the connection and why it defeats sqlite3 trace callbacks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: task-3012 gave AgentRunsDB a HELD per-thread connection cached on first use, so these three tests' monkeypatched _get_connection spy never ran (the fixture's own create_run calls had already opened the connection) and set_trace_callback never attached — the assertions saw zero statements. Repair is test-side only, production untouched: each test now calls db.close() after installing the spy so the next call reopens through it. Both surviving behavioral guards were mutation-tested live (BEGIN IMMEDIATE -> BEGIN DEFERRED failed the transaction test; COUNT(*) -> SELECT id + len() failed the materialize test), and the production file was diffed byte-identical to its pre-mutation backup afterward. 47 passed across the DB battery.
<!-- SECTION:NOTES:END -->
