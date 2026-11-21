---
id: TASK-42155
title: "Console Chat test rigs never shut their controllers down"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 22:23
updated_date: 2026-11-21 22:44
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
- [x] #3 The touched Chat files pass at -n 4 and on their lanes
<!-- AC:END -->

## Implementation Plan

1. Measure first: per-worker threads, fds, live controllers and `tldw-console-*` leftovers (private TMPDIR and basetemp) for the Chat files, with a throwaway probe plugin.
2. Add one shared fixture that shuts down every `ConsoleChatController` a test builds, and route `_reload_console`, `_durable_controller` (both files) and the controller test file through it.
3. Add a teardown check that fails when a test leaves a controller not shut down; show it red before the change.
4. Re-measure, run the touched files at -n 4 and their lanes, preflight and the ruff gates.

## Implementation Notes

A shared fixture now shuts down every Console controller these files build, and fails the test if one is left running. The one test in a touched file that was red at base is repaired too (coordinator decision).

**Fix (AC#1).** `Tests/Chat/conftest.py` gains `shut_down_console_controllers`. While the test runs it wraps `ConsoleChatController.__init__` (a scoped `MonkeyPatch.context()`, undone before teardown work starts) to record every controller built, by the test or by a rig. At teardown it awaits `shutdown()` on each one not yet disposed, newest first, then asserts none is left with `_disposed` False.
- The 5 s `wait_for` bound is soft: `shutdown()` absorbs cancellation, so a stuck shutdown runs on until pytest-timeout, which fails the test loudly. The product fix is the follow-up for a `shutdown()` that stops early, which the coordinator is filing.
- Six files apply it module-wide through `pytestmark = [bootstrap_profile, usefixtures("shut_down_console_controllers")]`:
  - `test_console_chat_controller.py` (210 constructions, 3 shut down by hand before)
  - `test_console_trace_discarded_tool_run.py` (`_reload_console`, plus the controllers `_project_console` builds there)
  - `test_console_prompt_queue_coordinator.py` (`_durable_controller`)
  - `test_console_rewind_summarize.py` (`_durable_controller`)
  - `test_console_trace_failed_tool_run.py` and `test_console_trace_surface_identity.py`, which import `_reload_console` (added after review rev1)
- No test or helper shuts down by hand.
- Decision: exactly one shutdown failure is reported as a `UserWarning`, not a failure: a `ValueError` whose text is `Resolve pending dispatch before changing conversation branches.`. Any other exception from `shutdown()` is collected and raised as a `BaseExceptionGroup` after every controller has been tried. The first version warned on every exception; review rev1 narrowed it. 26 tests deliberately leave a dispatch checkpoint pending: `test_discard_evidence_is_revalidated_before_dispatch[*]` and `test_each_discarded_followup_is_revalidated[*]` in the discarded-run file, and `test_controller_retry_reuses_exact_owned_compound_reservation[*-fresh]`, `test_two_saved_turns_keep_history_references_through_production_trace[changed-history]` and `test_controller_refuses_reserved_boundary_with_other_accepted_owner[fresh]` in the controller file.
  - For them, `_finish_begin_shutdown` abandons the session's preparation. That deletes its row, and the delete refuses with `ValueError: Resolve pending dispatch before changing conversation branches.` (`console_chat_store._dispatch_branch_transaction`).
  - Failing on it would have turned the Console P0 lane red for product behaviour this task does not own. It is follow-up 6 in TASK-42156's coordinator note: a controller `shutdown()` that stops early when `_finish_begin_shutdown` raises.
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

**Lanes and -n 4 (AC#3).**
- Lane runs, all green:
  - recovered-suites flags (serial, `--storage-lease-guard --timeout=180`) on `test_console_rewind_summarize.py` and the laned controller node: 74 passed in 126 s (load 30-56).
  - Console P0 admission flags (serial, `--timeout=300`) on `test_console_project_instruction_traces.py`, `test_console_trace_failed_tool_run.py` and `test_console_trace_discarded_tool_run.py`: 149 passed, 18 shutdown warnings, in 1,054 s (load 40-56).
- `test_console_prompt_queue_coordinator.py::test_accepted_queued_prompts_use_normal_persistence_exactly_once` was red at base, waiting out pytest-timeout on `await gateway.started[0].wait()`.
  - Cause: its `RecordingPersistence` double (unchanged since 2026-10-01) has no prompt mix-in storage. A durable send saves its mix-in attempt before the provider is entered, so `ConsolePromptMixinService.accept_attempt` raised `MixinSnapshotUnavailableError("mixin_persistence_unavailable")`. `_block_context_preflight` then blocked the turn, and the provider never started.
  - Fix (coordinator decision): the double gains a no-op `save_conversation_mixin_attempt` with the real signature, and a `db` flagged `is_memory_db`, so `_run_owned_save` runs the save inline. The test is the double's only user.
  - Red at base: `Failed: Timeout (>120.0s)` serially. Green with the fix: 1 passed in 1.7 s.
- After the fix, the four files at `-n 4`: **510 passed**, 26 warnings (the 26 refused shutdowns above), in 186 s (load 20-23).

**Review rev1.**
- **Fixture narrowed.** Only the exact pending-dispatch `ValueError` warns now; any other `shutdown()` error is raised in a `BaseExceptionGroup` after the loop. Negative control: a `RuntimeError` injected right after `begin_shutdown()` (temporary edit, reverted) made `test_console_rewind_summarize.py::test_untyped_memory_adapter_preserves_newest_first_order` fail at teardown with `ExceptionGroup: Console controller shutdown failed (1 sub-exception)`. The same node passes without the injection.
- **Two more files.** `test_console_trace_failed_tool_run.py` and `test_console_trace_surface_identity.py`, which import `_reload_console`, now apply the fixture.
- **Re-runs.** Narrowing surfaced no hidden shutdown error. Every warning is the pending-dispatch refusal: 61 of them, 35 from `test_console_trace_failed_tool_run.py`.
  - Seven files at `-n 4` (the six above plus `test_console_project_instruction_traces.py`), load 10-13: 637 passed, 64 warnings (the 61 refusals and 3 `fd_leak_sentinel` lines).
  - The Console P0 step's three trace files serially with its flags, load 13-23: 149 passed, 53 warnings.
- **Still open.** `_project_console` (`test_console_project_instruction_traces.py`, used by 6 files) still never shuts its controller down. The coordinator is filing it as a follow-up.

**Files:** `Tests/Chat/conftest.py`, `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_trace_discarded_tool_run.py`, `Tests/Chat/test_console_prompt_queue_coordinator.py`, `Tests/Chat/test_console_rewind_summarize.py`, `Tests/Chat/test_console_trace_failed_tool_run.py`, `Tests/Chat/test_console_trace_surface_identity.py`.
