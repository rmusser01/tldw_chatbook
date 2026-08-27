---
id: TASK-22720
title: Agent bridge placeholder-replacement test trips the unresolved-recovery guard
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


## Correction 2026-08-27 — the filed premise was wrong on both counts

Measured by instrumenting the controller's `except Exception` arm, not inferred.

**1. The controller does NOT swallow the failure.** That handler builds
`"Agent run failed: {describe_stream_failure(exc)}"`, calls
`mark_message_failed`, appends a failure system row, and sets run state FAILED.
The failure is surfaced, not lost. The title of this task is inaccurate.

**2. What `run_reply` raises, and why:**

    RuntimeError: Unresolved temporary dispatch recovery cannot be replaced.

Raised from `ConsoleChatStore.restore_state` (console_chat_store.py:5046). The
guard refuses to replace an UNRESOLVED dispatch recovery unless the restored
message set still contains the recovery's own USER and ASSISTANT rows. The
test's `_ReplacingBridge` deliberately builds `retained` with the assistant row
EXCLUDED -- that is how it simulates "the placeholder went missing" -- so it
trips the guard by construction.

The guard looks correct: silently replacing an unresolved recovery whose
assistant row has vanished would discard recovery state. So this is a stale TEST
TECHNIQUE, not a product defect: the double simulates a missing placeholder in a
way the store now forbids.

**Open question for whoever picks this up** (do not guess it):
what SHOULD happen when an agent replaces a placeholder that is gone? Either
  (a) the test clears/resolves the dispatch recovery before `restore_state`, so
      the simulation stops violating a real invariant; or
  (b) the expectation changes to the surfaced "Agent run failed: ..." outcome,
      if tripping the guard is the honest result of that scenario.

Pre-existing either way: this was one of the original 40 failures in
`test_console_local_citation_boundary.py`, failing before TASK-22301 touched it.
