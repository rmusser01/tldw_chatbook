---
id: TASK-22720
title: Agent bridge run_reply failure is swallowed, leaving an empty assistant row
status: To Do
labels:
  - console
  - agents
  - bug
priority: medium
---

## Description

`test_citation_repair_agent_missing_placeholder_keeps_runtime_row_without_repair`
fails with `assert '' == 'runtime replacement'`. This is PRE-EXISTING: it is one
of the original 40 failures in `test_console_local_citation_boundary.py`,
present before TASK-22301 touched the file, and it is the last one remaining.

Measured during TASK-22301, not inferred:

- The bridge's `run_reply` DOES run -- its `calls` list is non-empty, asserted
  as a harness precondition.
- It does NOT reach its own replacement code -- the `replacement_id` it records
  immediately after `append_message` is never set.

So `run_reply` raises partway through and the controller swallows the exception,
leaving the assistant row empty and the turn apparently "successful". A bridge
failure that silently produces a blank answer is the user-visible symptom.

The swallowing is the defect worth fixing; whatever `run_reply` trips over is
secondary and may well be a stale expectation in the test's own bridge double
(it calls `session_id_for_message`, `restore_state`, and a `_first` over
`sessions()`). Establish WHICH call raises before changing anything.

## Acceptance Criteria

- [ ] The exception raised inside `run_reply` is identified and named
- [ ] An agent-bridge failure is no longer silently swallowed into an empty
      assistant row -- it is either surfaced or logged with its exception type
- [ ] The xfail marker in `test_console_local_citation_boundary.py` is removed
      and the test passes
