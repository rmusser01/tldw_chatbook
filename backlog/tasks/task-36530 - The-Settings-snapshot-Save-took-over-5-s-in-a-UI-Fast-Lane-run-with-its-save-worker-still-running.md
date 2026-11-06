---
id: TASK-36530
title: The Settings snapshot Save took over 5 s in a UI Fast Lane run, with its save worker still running
status: Done
assignee:
- '@claude'
updated_date: 2026-11-06 00:25
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- settings
- ci-flake
priority: medium
references:
- Tests/UI/test_llamacpp_snapshot_settings.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/LLM_Management/snapshot_settings.py
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the CI run on PR #3225 (run 40024354185, UI Fast Lane (4), job 120656651965, 2026-10-23). `Tests/UI/test_llamacpp_snapshot_settings.py::test_f9_snapshot_controls_paint_and_keyboard_save[size1]` (140x45) is a UI census entry. It failed with `TimeoutError` at the 5 s `asyncio.timeout` around the save (:339-341), and so failed the required check. The lane runs the census serially, so the timeout did not come from xdist contention.

The failure note, added in c5897f76c4 after an earlier failure, shows the save was not lost but still running:
- before Enter: (focused, disabled, dirty, saving) = (True, False, True, False);
- after the timeout: (True, False, True, True);
- `loaded enabled: False`, `save workers: ['RUNNING']`.

So Enter reached the Save branch (settings_screen.py:33101-33116), but the worker's `asyncio.to_thread(save_snapshot_preferences, ...)` (:9392) had not returned after 5 s. `save_snapshot_preferences` is one `apply_settings_mutation_to_cli_config` call with a locked precondition (snapshot_settings.py:38-66).

On dev 5b5920b8e2 locally, a temporary timing probe (since reverted) measured that call 8 times, 4 of them in concurrent loaded runs at load average about 13. Each took 0.14-0.19 s, with no executor queue wait. A 5 s stall is about 30 times that, so this does not look like a wait that is too short for normal load. The cause is not identified. Candidates: the default executor queue, the config owner's lock, or the write itself on the CI runner.

**More occurrences (recorded 2026-11-02).** Searching 62 failed UI-job logs from 46 derived-artifacts runs (2026-10-21 to 11-02) found four more failures. Each was `[size1]` with a child `TimeoutError`, in UI Fast Lane (25):

| Run | Attempt | Branch | Date | Outcome |
|---|---|---|---|---|
| 40909862501 | 1 | codex/task-35582-delivery | 2026-10-29 | green on rerun |
| 41003666368 | 1 | fu/h-railstate (#3225) | 2026-10-30 | green on rerun |
| 41189968143 | 1 | fu/h-railstate (#3225) | 2026-10-31 | green on rerun |
| 41224394415 | 1 | fu/h-railstate (#3225) | 2026-10-31 | stayed red |

The same set holds 19 UI (25) logs: the 4 above failed this node and 15 passed it. Of the 16 UI (25) jobs on other branches, 1 failed it. `[size0]` is not a census entry. Four of the five known failures, counting the first, came from #3225's branch, which changed Console rail-state persistence (TASK-36385). On 2026-10-31 that branch failed the node in 2 of 4 attempts:
- 41189968143: attempt 1 failed, attempt 2 passed;
- 41224394415: attempt 1 failed;
- 41248007404 (the merge-time run, on d3dc1c99b0): attempt 1 passed, with a 20.14 s call.

A local paired probe found no difference that #3225 caused:
- The branch added no config writes to this test.
- The snapshot save took 147-235 ms on the branch and 131-275 ms on dev.
- The test passed 4 of 4 runs on both trees with all 18 CPUs loaded.
- The branch's UI (25) shard ran 600 s, inside the 528-814 s of other runs.

The owner finished #3225 with three more commits (retained rail seeds: b787009712, d02046bf61 and edd6260383, which are 3098e0705c, 71ddbecdbb and d3dc1c99b0 on dev) and merged it on 2026-10-31. The final code ran twice in CI: 41224394415 on edd6260383 failed, and the merge-time run passed. No run on dev since then has been checked. If it fails again, instrument it in CI: log the save thread's executor wait and lock wait from a temporary debug commit. Local runs have not reproduced it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test's failure note says where the save's time went (executor queue, lock wait, write, resume)
- [x] #2 The cause of the CI stall is identified and recorded; if Save can be held behind other work in the app, that is fixed or recorded as accepted with a reason
- [x] #3 The test passes 20 of 20 serial runs, and its time bound rests on measured save times
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test-only timeline (monkeypatch, no production cost): stamp the worker start, save-thread start/end and config write-lock request/acquire, and on timeout name the phase still running, with the default executor's load and the lock owners' innermost frames.
2. Read the five failed CI logs for the save's phase; probe locally with the default executor capped at the CI size (4 vCPU -> 8 threads) and dump what holds its threads at Enter.
3. If Save queues behind other app work on a shared executor or lock, fix it where every caller routes through; otherwise record it as accepted with the reason.
4. Run the test 20 times serially, record the phase distribution and load, and set the bound from it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (AC#2).** The save thread was short of the GIL. Nothing was queued in front of it. The test waited with `while not enabled: await pilot.pause(0.01)`. Each `pilot.pause()` walks all 477 Settings nodes on the event loop and posts a callback to each one. Measured here, a pause cost 13-15 ms of loop CPU per 10 ms, so the loop was busy about 55% of the time, and more on a slower CI core. The config write makes hundreds of syscalls, at storage-admission directory pins and descriptor opens, and gives up the GIL at each one. To get it back, it waits out the busy thread's switch interval.
- One save on a thread with an idle main thread: 0.08-0.10 s.
- The same save with a main thread spinning in Python: 2.6-11.8 s at the 5 ms switch interval, 2.3-3.1 s at 1 ms, 0.8-1.2 s at 0.2 ms.
- A CPU-bound loop in the mounted test reproduced the CI note exactly: `(True, False, True, True)`, `loaded enabled: False`, worker RUNNING. It was in the write phase, with 10.2 s of loop CPU in 10.5 s.
- With the save delayed until the wait was running, the old poll used 0.42-0.52 s of loop CPU and the write took 0.25-0.36 s. Awaiting an event used 0.04-0.05 s and the write took 0.10-0.18 s.

The CI logs match this. In all four failure logs with teardown output, the FTS backfill's ChaChaNotes creation had started during the call and finished only in teardown. That creation takes 0.4 s on CI, so it was starved by the same busy loop and finished once the poll stopped. TASK-38462 has since stopped the backfill creating a missing database.

Other causes ruled out:
- The default executor (8 threads on a 4-vCPU runner) had 0 threads busy at Enter in every probe, including an 18 s delay with the pool capped at 8.
- The config locks were free at Enter. A concurrent ChaChaNotes creation left the save at 0.14 s.

**Fixed in the test.** The wait now awaits an `asyncio.Event` that the save worker sets, and does not poll. App work does not hold up Save: without the poll, the loop is idle during the save.

**Accepted with a reason.** Any thread running CPU-bound Python can stretch a config write by 25-140x through GIL hand-offs. Removing that cost means fewer GIL-releasing calls per storage-admission operation, or a scoped switch interval (the precedent is `Chat/console_preparation_reads._preparation_handoffs`). Either change touches every config write, which is outside this flake. It is listed as a proposed follow-up.

**Instrumentation (AC#1).** `_SaveTimeline` is test-only and works through monkeypatch, so production code carries no instrumentation. It wraps `save_snapshot_preferences`, `config._config_write_lock` and the screen's `_save_snapshot_preferences_draft`. The timeout note gives:
- the time of each phase: dispatch, executor queue, lock wait, write, resume;
- which phase is still running, and the save thread's innermost app frame;
- the loop's CPU since Enter against the wall time;
- how many default-executor threads are busy and how many jobs are queued;
- who owns the rebuild lock and the file lock;
- the save thread's CPU, on Linux.

Three negative controls, each a temporary test edit that was then reverted, each gave the matching reading:
- CPU-bound loop: `write still running at 1.97 s ... event loop CPU since Enter: 10.23 s of 10.54 s`;
- full executor: `executor queue still running at 5.07 s ... 22/22 busy, 3 queued`;
- held lock: `lock wait still running at 5.06 s ... rebuild lock held by neg-holder`.

**Bound (AC#3).** There were 20 serial runs of the test, both sizes, so 40 saves, with no xdist. Load average was 8.9-14.4 on 18 cores, shared with other agents. Result: 20/20 runs passed, 40/40 saves.

Phases were recorded by a temporary out-of-test plugin through `PYTEST_PLUGINS`, deleted afterwards, so the test code was the final version.

| Phase | Median | p90 | Max |
|---|---|---|---|
| Executor queue | 0.000 s | 0.000 s | 0.000 s |
| Lock wait | 0.009 s | 0.012 s | 0.025 s |
| Write | 0.106 s | 0.150 s | 0.266 s |
| Resume | 0.016 s | 0.020 s | 0.022 s |

The wait after Enter took 0.093-0.296 s, median 0.135 s. The 5 s bound stays: it is 17 times the local maximum, and still about 5 times that maximum if a CI core runs 2-3 times slower.

**Pre-existing reds.** The 13 plain (non-private-profile) tests in this file fail locally with `RecoveryRequired` (TASK-38111). The same 13 nodes fail at base 29592ecc9b with the same command.

**Files:** `Tests/UI/test_llamacpp_snapshot_settings.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
