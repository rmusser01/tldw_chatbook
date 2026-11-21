---
id: TASK-42156
title: "A full local Tests/Chat run at -n 4 hangs at the end with a large fd and thread growth"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-20 22:23
updated_date: 2026-11-21 21:41
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/conftest.py
- Tests/Chat/
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A consolerun follow-up (round 18). It is reported from that stream's full local runs and was not rerun here, because a full `Tests/Chat` run takes over 15 minutes.

A full local `Tests/Chat` run at `-n 4` hangs at the end. One worker sits idle for about 15 minutes, `fd_leak_sentinel` (`Tests/conftest.py:717`) reports +1,501 file descriptors, and each worker holds 50-180 threads.

The controllers the Console rigs never shut down (filed separately) are a likely contributor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The hang is reproduced, or shown absent, with the run's command, load and timings recorded
- [x] #2 The cause of the end-of-run idle worker is named from stack or fd evidence, not inferred
- [ ] #3 Either the hang and the fd growth are fixed, with fd_leak_sentinel's growth recorded before and after, or a task is filed for each cause found
<!-- AC:END -->

## Implementation Plan

1. Run Tests/Chat at -n 4 through the wrapper with --timeout=300 and a throwaway probe plugin that dumps every worker's thread stacks when it is idle past the timeout, kills it after a bound, and records fds, threads and live controllers at session end.
2. Name the cause from the stacks and fd records.
3. Fix it if it is test infrastructure or leaked rigs; otherwise record the evidence and propose a follow-up.

## Implementation Notes

The idle time does not come from Console controllers. It comes from a few tests that wait out pytest-timeout, and from executor threads in `legacy_hook_ownership.wait_closed`, which has no timeout. The fd growth is SQLite databases tests never close, and the thread growth is storage-admission hold threads. None of these is test infrastructure or one of the controller rigs, so nothing is fixed here; each cause is listed below as a follow-up for the coordinator to file. AC#3 stays open until they are filed.

**Run (AC#1).**
- Command: `run.sh <wt> r19cb-chatbase python <probe runner> --basetemp=<wrapper bt> Tests/Chat -n 4 --timeout=300 -q --tb=line -rfE`, from dc371f0842 with a private `$TMPDIR`. The tree changed during the run only in `maintenance_drain` (TASK-42152) and in files no running test re-collects.
- The probe was a throwaway plugin, never committed. It re-armed a faulthandler stack dump whenever a worker started no test for 330 s, killed a worker idle for 1,200 s, snapshotted threads, fds and `tldw-console-*` directories at session end, and dumped stacks 120 s after `pytest_unconfigure` and killed the process at 300 s.
- 18:24-21:40Z, load 13-56 (other sessions running). Result: 843 failed, 22,517 passed, 147 skipped, 14 errors in 11,641 s (3 h 14 min).
- **No end-of-run hang in this run.** All four workers reached `pytest_unconfigure` at 13:37:46-56 local and exited within 2 min. No post-unconfigure dump and no kill.
- **The idle-worker symptom did appear mid-run.** Workers sat at 0% CPU through:
  - four teardowns of 597-902 s: `test_hooks_v2_teardown.py::test_mounted_stop_button_cancels_pending_stop_proposal[size1-False]` (901.6 s) and `[size1-True]` (601.9 s); `test_hooks_v2_continuations.py::test_consumption_before_foreground_or_uncertain_failure_is_not_reopened[stop]` (598.8 s) and `::test_lost_scheduler_response_never_returns_machine_work_to_queue` (596.5 s);
  - 12 calls that ran into the 300 s timeout: six in `test_console_turn_library_authority.py`, two in `test_console_turn_execution_context.py`, and one each in `test_console_voice_preflight.py`, `test_console_hook_preparation_lifetime.py`, `test_console_prompt_queue_coordinator.py` and the `[size1-False]` node above.
  - gw1 alone was idle from 12:44 to about 13:15 local in the two `[size1-*]` nodes, longer than the "about 15 minutes" the report describes.
- **The exit hang, reproduced in isolation.** The four params of `test_consumption_before_foreground_or_uncertain_failure_is_not_reopened`, serially with `--timeout=60` (load about 40): `3 passed, 1 error in 376.71s`. `[stop]`'s teardown timed out. The process then never exited: `pytest_unconfigure` ran at 12:41:26, and the process was still alive at 12:50, when a bounded 15 min watchdog killed it.

**Cause (AC#2), from stacks.**
- In the isolated repro, the dump 100 s after unconfigure showed:
  - the main thread in `threading._shutdown` → `concurrent.futures.thread._python_exit` → `join`, which `sample` shows as `Py_FinalizeEx` → `wait_for_thread_shutdown`;
  - an executor thread in `Agents/legacy_hook_ownership.py:40 wait_closed`, an unbounded `Condition.wait_for` reached through `asyncio.to_thread`;
  - asyncio's `_do_shutdown` thread joining that executor.
- In the full run the dumps of the stuck workers showed the same executor thread (gw1 at 12:33, gw0 at 12:58 local). At session end, gw0 and gw1 still held non-daemon `asyncio_0` and `_do_shutdown` threads.
- The mechanism is TASK-41028's (`console_runtime._drain_legacy_hooks` and `hooks_v2/engine.py` wait on `wait_closed` with no bound and absorb cancellation). Here it is reached from four hooks_v2 nodes. pytest-timeout frees each teardown stage only after another 300 s (TASK-41135's re-arm), and a process whose stuck thread is still alive at exit cannot exit.
- Why this run ended cleanly although gw0 and gw1 still held those threads at session end was not established: the threads ended within 2 minutes of unconfigure.

**fd and thread growth** (`fd_leak_sentinel` per worker, and `lsof`/`ps -M` on the live workers):
- `fd_leak_sentinel`: +1,237, +1,652, +1,818 and +2,054 per worker (start 17).
- Mid-run (`lsof`, about 50%): 1,050-1,666 fds per worker. 733-1,284 of them are SQLite files of databases tests never close, 4-8 fds per database file. `runs.db` alone accounts for 170-495 per worker; `agent_runs.db`, `chacha.sqlite`, `workspaces.sqlite` and `controller.sqlite` follow. The rest are mostly one lock file plus one directory per storage hold.
- Threads at session end: 87-217 per worker, of which 84-212 are `chatbook-storage-admission` hold threads. `storage_admission._Hold` keys a hold by `(pid, bootstrap root)`, and the per-test sandbox gives every test its own root. So each sandboxed test that leaves a storage lease open (an unclosed database) leaves a daemon hold thread behind. These threads do not block exit.
- The controllers TASK-42155 shuts down are not a factor. On the four controller-heavy files, shutting every controller down changed neither the threads (3 per worker before and after) nor the fd range. Live controllers at session end were 0-9 per worker.
- Scratch: `$TMPDIR` still received 30 `tldw-console-*` directories during the run. They were created while the `test_workflow_*` native files ran (birth times against each worker's test starts), so TASK-42112's basetemp default does not reach them. The rest stayed in basetemp (193-634 per worker).

**Not fixed here (AC#3).** These are product custody or many separate test rigs, not test infrastructure, so they go to the coordinator to file: the four hooks_v2 `wait_closed` nodes (or into TASK-41028), the 300 s timeouts not covered by TASK-40638, the storage-admission holds and SQLite handles tests leave open, and the 30 `$TMPDIR` scratch directories. AC#3 is unticked until those are filed.

**Files:** this task file, and a lessons entry in `backlog/docs/lessons-testing-evidence.md`. No code changed.
