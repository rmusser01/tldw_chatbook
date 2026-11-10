---
id: TASK-39168
title: "Two goal-dispatch tests bound the provider's arrival by 3 s of wall clock, so they fail on a loaded host"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:56
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/Chat/test_console_goal_dispatch.py
dependencies:
- TASK-39143
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (lanes and cprod streams). Checked by reading on dev b590a15afd, 2026-11-08. Line numbers are dev's.

**The two tests.** `test_repeated_start_cannot_reprovision_an_accepted_attempt` (`:373`) and `test_goal_authorization_cannot_be_forged_or_moved_to_another_session` (`:408`) each hold a threading provider.
- The provider sets `entered` and blocks on `assert release.wait(3)` (`:383`, `:420`).
- The test asserts `await asyncio.to_thread(entered.wait, 3)` (`:391`, `:437`).

**What failed.** The lanes stream saw both red on dev 8b762b52fa under load, where `entered.wait(3)` returned False. They passed 3 of 3 at load 37-43, and passed here at load ~21.

**The precedent.** TASK-39143 (round 12, fu/r12-ctlr) replaced the same shape in the controller file with `until_provider_starts`. That helper waits on the provider's start, and fails naming the turn's result if the turn ends first.

**Lanes.** No PR lane runs the file, on dev or on the stack. Round 12 (fu/r12-cprod, TASK-39119) adds nodes to it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both tests wait on the provider's start or the turn's end, not a 3 s wall-clock bound, and fail naming the turn's result when the turn ends first; the provider's own `release.wait(3)` cannot fail a run that otherwise passes
- [x] #2 Both pass 20 of 20 runs at `-n 4`, with load recorded, at least ten of them at a one-minute load of 40 or more
- [x] #3 The file runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Both tests: the provider sets an asyncio event through `loop.call_soon_threadsafe` and the test waits with `until_provider_starts` (provider start or the turn's end, naming the turn's result).
2. The provider's `release.wait` loses its 3 s bound and its assert; every path to the test's end sets `release` in a `finally`.
3. Red: show the old 3 s wait failing when the provider start is delayed past it, and the new wait failing naming the turn's result when the turn ends first.
4. 20 runs at -n 4 with load recorded, ten or more at load >= 40.
5. Lane: the whole file replaces its three nodes in the "PR lane (recovered suites)" step; update the step note and argv pins; three timings of the step.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base dev ae9f9e8d21, 2026-11-10. Test-only change, plus the lane.

**Waits (AC#1).** Both tests' providers now set an `asyncio.Event` through `loop.call_soon_threadsafe`, and the tests wait with `until_provider_starts(entered, task)` (`Tests/console_provider_doubles.py`, TASK-39143's helper): the provider's start, or the dispatch task's end, which fails naming its result. The provider's hold lost its 3 s bound and its `assert`: only the first call holds, and the test's `finally` releases it. A regression that calls the provider a second time (the "manual alias" send in the forgery test) therefore returns at once and fails on its own assertion, instead of hanging on an unbounded hold.

**Red evidence** (temporary copies of the file, `-n 4`):
- dev's file, unedited, whole file 3 times at load 46-65: run 3 failed both nodes at `assert await asyncio.to_thread(entered.wait, 3)` (`assert False`), 2 failed / 18 passed. A variant run at load 73 failed both nodes on the same line.
- dev's file with the provider's start delayed 3.5 s: both nodes failed on the same line. The new file with the same delay: 2 passed.
- dev's file with the 3 s wait unbounded and a 3.5 s pause after the provider started: the reprovision test failed with `AttributeError: 'NoneType' object has no attribute 'context'`, because the provider's `assert release.wait(3)` ended the attempt. The new file with the same pause: 2 passed.
- The new file with `_apply_skill_substitution` refusing the turn: both failed with `AssertionError: ended before the provider started: GoalIterationResult(... native_run_id=None ...)`.

**Runs (AC#2).** The whole file (20 tests) 20 times at `-n 4`, two runs at a time: 20 of 20 passed, every run at a one-minute load of 55-92 (all 20 at 40 or more), 40.8-52.3 s each.

**Lane (AC#3).** The whole file replaces its three TASK-39119 reply-pin nodes in "PR lane (recovered suites)"; its module `pytestmark` is `bootstrap_profile`, and it keeps their place among the bootstrap_profile suites. Guard-clean with the library preparation file (TASK-39163's notes). Never pr-fast-lane.

**Step timings (AC#3, shared with TASK-39163/39168).** `jobstep.sh fu-r13-load "Run the recovered suites"` (19 targets, serial, `--storage-lease-guard --timeout=180`), 2026-11-10, each 839 passed, 1 skipped, no lease finding:

| Run | pytest time | Load (1 min) |
|---|---|---|
| 1 | 1444.4 s (24.1 min) | 67-78 |
| 2 | 943.3 s (15.7 min) | 36-67 |
| 3 | 713.2 s (11.9 min) | 36-37 |

Run 1 is over the job's 20-min cap, at a load of 67-78 on 18 CPUs (four streams of other agents' runs plus this task's 20 goal-dispatch runs ending as it started). The additions alone, serially under the guard: the goal-dispatch file 94.7 s (load 36-39; it replaces three nodes already in the step) and this library file 143.2 s (load 39-63), about 210 s of run 3's 713 s. CI ran this step in 5.2-5.5 min (job 6.0-6.5 min) on the three latest PR runs that finished it (42841528703, 42841511400, 42737620260, read-only `gh`). Charging the additions at the same share (about 30%) projects about 7.8 min of step and 9 min of job on CI, 11 under the cap; a PR run of this branch is needed to confirm it.

**Not done here.** `test_both_coordinators_wait_for_one_runtime_audit` and `test_cancellation_during_preparation_retains_primary_until_own_cleanup` in the same file still bound a start by 2 s (`to_thread(entered.wait, 2)`, `wait_for(entered.wait(), 2)`); they passed every run above and are outside this task's two tests.

**Files.** `Tests/Chat/test_console_goal_dispatch.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
<!-- SECTION:NOTES:END -->
