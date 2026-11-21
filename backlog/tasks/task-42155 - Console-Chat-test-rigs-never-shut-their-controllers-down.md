---
id: TASK-42155
title: "Console Chat test rigs never shut their controllers down"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-20 22:23
updated_date: 2026-11-21 19:46
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_chat_controller.py
- Tests/Chat/test_console_trace_discarded_tool_run.py
- Tests/Chat/test_console_prompt_queue_coordinator.py
- Tests/Chat/test_console_rewind_summarize.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A consolerun follow-up (round 18), checked against the code:
- `Tests/Chat/test_console_chat_controller.py` constructs `ConsoleChatController(` 210 times and awaits `controller.shutdown()` 3 times.
- The rig helpers `_reload_console` (`test_console_trace_discarded_tool_run.py`) and `_durable_controller` (`test_console_prompt_queue_coordinator.py`, `test_console_rewind_summarize.py`) build controllers without shutting them down; a few tests shut theirs.

`ConsoleChatController.shutdown` exists (`Chat/console_chat_controller.py`). Since TASK-42112 the controllers' scratch directories are contained in pytest's basetemp. The controllers' threads, connections and other resources still outlive their tests. That likely feeds the end-of-run hang and fd growth filed separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Chat rig helpers and the controller tests shut down every controller they create, through a fixture or the helper itself, not by hand in each test
- [x] #2 A check fails when a test leaves a controller running (for example a count of live controllers or their worker threads at teardown), red before the change on at least one test
- [ ] #3 The touched Chat files pass at -n 4 and on their lanes
<!-- AC:END -->

## Implementation Plan

1. Measure first: per-worker threads, fds, live controllers and `tldw-console-*` leftovers (private TMPDIR and basetemp) for the Chat files, with a throwaway probe plugin.
2. Add one shared fixture that shuts down every `ConsoleChatController` a test builds, and route `_reload_console`, `_durable_controller` (both files) and the controller test file through it.
3. Add a teardown check that fails when a test leaves a controller not shut down; show it red before the change.
4. Re-measure, run the touched files at -n 4 and their lanes, preflight and the ruff gates.

## Implementation Notes

A shared fixture now shuts down every Console controller these files build, and fails the test if one is left running. AC#3 is left open: one test in a touched file, red at base, still fails. See below.

**Fix (AC#1).** `Tests/Chat/conftest.py` gains `shut_down_console_controllers`. While the test runs it wraps `ConsoleChatController.__init__` (a scoped `MonkeyPatch.context()`, undone before teardown work starts) to record every controller built, by the test or by a rig. At teardown it awaits `shutdown()` (5 s bound) on each one not yet disposed, newest first. Then it asserts none is left with `_disposed` False.
- The four files apply it module-wide through `pytestmark = [bootstrap_profile, usefixtures("shut_down_console_controllers")]`:
  - `test_console_chat_controller.py` (210 constructions, 3 shut down by hand before)
  - `test_console_trace_discarded_tool_run.py` (`_reload_console`, plus the controllers `_project_console` builds there)
  - `test_console_prompt_queue_coordinator.py` (`_durable_controller`)
  - `test_console_rewind_summarize.py` (`_durable_controller`)
- No test or helper shuts down by hand.
- Decision: a `shutdown()` that raises after it has fenced the controller is reported as a `UserWarning`, not a failure. 26 tests deliberately leave a dispatch checkpoint pending: `test_discard_evidence_is_revalidated_before_dispatch[*]` and `test_each_discarded_followup_is_revalidated[*]` in the discarded-run file, and `test_controller_retry_reuses_exact_owned_compound_reservation[*-fresh]`, `test_two_saved_turns_keep_history_references_through_production_trace[changed-history]` and `test_controller_refuses_reserved_boundary_with_other_accepted_owner[fresh]` in the controller file.
  - For them, `_finish_begin_shutdown` abandons the session's preparation. That deletes its row, and the delete refuses with `ValueError: Resolve pending dispatch before changing conversation branches.` (`console_chat_store._dispatch_branch_transaction`).
  - Failing on it would have turned the Console P0 lane red for product behaviour this task does not own (proposed follow-up below).
  - `begin_shutdown` sets `_disposed` before it raises, so the running check still holds for them.

**Check (AC#2), red first.** With the fixture's shutdown loop removed (temporary edit, reverted), the four files at `-n 4` gave 509 passed, 1 failed and **430 teardown errors**, all `AssertionError: N Console controller(s) left running`: 304 in the controller file, 62 in rewind/summarize, 33 in the prompt queue file and 31 in the discarded-run file. With the loop: no errors.

**Measurement** (`-n 4` through the wrapper with a private per-run `$TMPDIR`; a throwaway probe plugin, never committed, recorded per worker at session end):

| Run | Load | Result | `tldw-console-*` in `$TMPDIR` | in basetemp | threads per worker |
|---|---|---|---|---|---|
| Before (base) | 47-49 | 509 passed, 1 failed, 477 s | 0 | **356** | 3 |
| After | 33-46 | 509 passed, 1 failed, 26 warnings, 493 s | 0 | **33** | 3 |

- TASK-42112's conftest default already keeps every directory out of `$TMPDIR`. The 33 left in basetemp come from the 26 shutdowns that refused (their `shutdown()` stops before it disposes scratch) and from the hung test below.
- These files leak no threads, before or after: 3 per worker at session end (main, execnet, storage admission), plus the probe's own. Live controllers at session end were 0-1 per worker both times.
- fd growth per worker stayed in the same noisy range (+18..+68 before, +48..+84 after), so the fd growth is not these controllers (see TASK-42156).

**Lanes and -n 4 (AC#3, not met).**
- Lane runs, all green:
  - recovered-suites flags (serial, `--storage-lease-guard --timeout=180`) on `test_console_rewind_summarize.py` and the laned controller node: 74 passed in 126 s (load 30-56).
  - Console P0 admission flags (serial, `--timeout=300`) on `test_console_project_instruction_traces.py`, `test_console_trace_failed_tool_run.py` and `test_console_trace_discarded_tool_run.py`: 149 passed, 18 shutdown warnings, in 1,054 s (load 40-56).
- At `-n 4` the four files give 509 passed, 1 failed. The failure, `test_console_prompt_queue_coordinator.py::test_accepted_queued_prompts_use_normal_persistence_exactly_once`, fails identically at base (same command: 509 passed, 1 failed). It is on no lane.
  - Alone it waits out pytest-timeout on `await gateway.started[0].wait()`. A temporary probe showed why: its `RecordingPersistence` double has no prompt mix-in storage, so `accept_attempt` raises `MixinSnapshotUnavailableError`. `_block_context_preflight` then blocks the turn ("Mix-in attempt could not be saved"), and the settlement reports "Dispatch terminal settlement previously failed.". The provider never starts.
  - The double has not changed since 2026-10-01. Repairing it is outside this task (proposed follow-up), so AC#3 stays unticked.

**Files:** `Tests/Chat/conftest.py`, `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_trace_discarded_tool_run.py`, `Tests/Chat/test_console_prompt_queue_coordinator.py`, `Tests/Chat/test_console_rewind_summarize.py`.
