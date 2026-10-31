---
id: TASK-37654
title: Console composer restyle acceptance test syncs the action row twice intermittently
status: Done
assignee:
  - '@claude'
created_date: 2026-10-30 08:10
labels:
- console
- flaky-test
priority: medium
references:
- backlog/tasks/task-33620.15.8 - Console-after-acceptance-the-composers-class-sync-and-the-durable-turn-publication-hold-run-starts-longest-UI-loop-callback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_composer_state_restyle.py::test_the_acceptance_projection_clears_now_and_restyles_send_after fails intermittently on dev with 'The projection should sync the action row exactly once, after its step, from the cleared draft: ['', '']'. It failed in PR #3286's UI Fast Lane (6) on 2026-10-30 and in 2 of 3 local runs on dev ff815f8559, so it is not caused by that PR. A flaky required-lane test evicts queued PRs, so either the projection double-syncs in a real race or the test's timing assumption is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the double action-row sync is identified as a product race or a test timing assumption, with evidence.
- [x] #2 The test passes 20 consecutive local runs on dev and does not weaken the exactly-once assertion unless the product contract changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Loop the test (single, whole file, xdist, two concurrent copies) to get a failure rate on dev.
2. Record a stack and the store draft/revision at every action-row sync and every store draft write after the projection.
3. Compare the test's setup with the product's acceptance path (`commit_session_input_draft` then `project_received_input`).
4. Fix the cause (product if the product double-syncs, else the test's setup) without weakening the exactly-once assertion.
5. Prove it: deterministic interleave old vs new, product mutations, 20 consecutive runs, whole file under load.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Cause: a test timing assumption (a faked store write), not a product race.**
The test faked the acceptance's input CAS by patching `store.session_draft` to return `""`. The store kept the typed draft at revision 1, and the projection recorded revision 1 as visible. An instrumented failing run showed what happened next. A Console sync tick (`_sync_native_console_chat_ui` -> `_sync_console_session_draft`) wrote the composer's `""` back to the store, moving it to revision 2. The next tick saw revision 2 != visible 1 and reloaded the composer with an undeferred `load_draft("")`: the second action-row sync. The second sync's stack was in `session.py` `_sync_console_session_draft` -> `load_draft`. It failed only when `pilot.pause()` (which waits up to 1 s for idle) lasted long enough for two ticks: about 0.26 s in failing runs, 0.04-0.06 s in passing ones.
In the product, both projection call sites (`console_received_dispatch.py`) run `store.commit_session_input_draft(...)` first. That clears the store draft and bumps its revision before the projection reads it, so the tick's write-back changes nothing and no reload follows.

**Fix (test only, `Tests/UI/test_console_composer_state_restyle.py`).**
- The test runs the real `commit_session_input_draft(session_input_snapshot(...))` right before the projection, as run start does.
- It then drives the tick's draft reconcile twice by hand before the pause, so the interleave that used to flake is now checked every run.
- The exactly-once assertion is unchanged.

**Evidence.**
- Before the fix, on dev 94a8a85f03, the test failed 7 times in 43 runs: 2 of 3, 2 of 4 and 1 of 10 single-test serial runs, and 2 of 16 whole-file `-n 2` runs with two copies running at once.
- Deterministic interleave: with the old fake read, the two hand-driven tick reconciles failed 3 of 3 (`A sync tick resynced the row: ['']`). With the real CAS they passed 3 of 3.
- Product mutations against the fixed test all go red:
  - no `defer_action_state`: "Send was restyled inside the acceptance projection";
  - the deferred sync dropped: `[] == ['']`;
  - the deferred sync run twice: `['', ''] == ['']`.
- After the fix:
  - 20 of 20 consecutive single-test serial runs passed;
  - 20 of 20 whole-file `-n 2` runs passed with two copies running at once;
  - 5 of 5 whole-file runs passed under `taskpolicy -b` (efficiency cores only).

A lesson was added to `backlog/docs/lessons-testing-evidence.md` under the "presentation edge" entry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
