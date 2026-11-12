---
id: TASK-39163
title: "test_console_automatic_library_preparation.py still has 14 unbounded waits on pre-provider boundaries, so a turn that ends early hangs the file until the 300 s timeout"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:56
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_automatic_library_preparation.py
dependencies:
- TASK-39045
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39045 on round-12 branch fu/r12-ctests; its notes propose this follow-up). Checked on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are the stack's, since TASK-39045 changes this file.

**The helper.** TASK-39045 added `_until_provider_starts(started, turn, timeout=30)` and moved every provider-start wait onto it. If the turn ends first, the helper fails naming the turn's result.

**The 14 waits still bare.** These waits on boundaries before the provider are still a bare `await ….wait()`:
- `service.entered` (`:1239`);
- `held.entered` (`:2444`, `:2486`, `:2549`, `:2711`, `:2754`, `:2879`, `:2962`, `:3170`, `:3260`, `:3406`, `:3493`);
- `boundary.entered` (`:2777`);
- `gateway.both_entered` (`:2820`).

**What happened.** TASK-39045's control was a temporary edit that made `_apply_skill_substitution` refuse every turn. Under it, the whole-file run hung on these waits until it was stopped.

**Lanes.** The file has 90 tests, 23.6 s at `-n 4` and load 24-34 (TASK-39045). It runs on no PR lane.

**Fixed elsewhere.** The vision-seam red that TASK-39045 also noted (`Tests/Chat/test_console_chat_controller.py::test_submit_draft_sends_image_parts_when_vision_capable`) is fixed by TASK-39138 in round 12. It is red on dev and passed on the stack (run.sh, load ~23).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every boundary wait in the file is bounded and, when the turn ends first, fails naming the turn's result
- [x] #2 Under the refuse-every-turn control, the whole file finishes with each such test failing on its bounded wait, well inside the 300 s timeout
- [x] #3 The file runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Generalise `_until_provider_starts` into `_until_entered(event, *turns, what=..., timeout=30)`; the provider helper calls it unchanged for its callers.
2. Move the 14 bare boundary waits (`service.entered`, `held.entered`, `boundary.entered`, `gateway.both_entered`) onto it, each naming its boundary; `both_entered` fails if either turn ends first.
3. Control: a temporary edit making `_apply_skill_substitution` refuse every turn; run the whole file, check it finishes well inside 300 s with each affected test failing on its bounded wait; revert.
4. Lane: add the whole file to the "PR lane (recovered suites)" step among the bootstrap_profile suites (module pytestmark), after showing it guard-clean serially; update the step note and the argv pins in Tests/CI; three timings of the step with jobstep.sh.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base dev ae9f9e8d21, 2026-11-10. Test-only change, plus the lane.

**Waits (AC#1).** `_until_provider_starts` is now a thin call to `_until_entered(entered, *turns, what=..., timeout=30)`. It waits for a boundary's `entered` event, and fails at once naming the first turn that ends (its result, exception or "cancelled"), or after 30 s if neither happens. All 14 listed waits use it, each naming its boundary ("the held preparation seam", "the held evidence capture", ...). `gateway.both_entered` passes both turns, so either one ending first fails the wait. A 15th bare wait the filing did not list was found by the control below: `test_live_loop_shutdown_cancels_and_awaits_submit_without_asyncio_diagnostics` awaited its own `held.wait()` (an `asyncio.Event` its history hold sets) on a private loop; it uses the helper too. The file's other waits are a bounded `asyncio.wait(..., timeout=5)` on the closed loop, a `threading.Barrier`, and the doubles' own holds.

**Control (AC#2).** A temporary autouse fixture in this file made `ConsoleChatController._apply_skill_substitution` refuse every turn (the filing's control, kept in the test file so parallel runs elsewhere in the worktree were not affected), at `-n 4`:
- With only the 14 listed waits bounded: 67 failed / 23 passed in 340.7 s, because the unlisted `held.wait()` hung its test to the 300 s timeout (300.17 s call). That is the red.
- With all 15 bounded: 67 failed / 23 passed in 66.6 s (load 56-76), and again in 85.2 s (load 86-106). Every boundary test whose seam comes after skill substitution failed on its bounded wait with `Failed: the turn ended before <boundary>: ConsoleSubmitResult(accepted=False, ... visible_copy='r13d control: refused before the provider' ...)`, 18 nodes, three of them in private-profile children; the two `test_recovered_queue_acknowledges_postaccept_cancellation_once` nodes failed earlier on `_paused_queued_send`'s provider wait.
- Three boundaries come before skill substitution (`service.entered`, `boundary.entered`, `both_entered`), so that control reaches them. A second temporary edit made each double raise before setting `entered`: all three failed at once naming the result (`ConsolePreparationOutcome(...)`, `RuntimeError('r13d control: boundary refused')`, `RuntimeError('r13d control: resolve refused')`), 3 failed in 13 s.
Both controls were reverted; the file then passed 90 of 90 at `-n 4` (70.6 s, load 37-80).

**Lane (AC#3).** The file joins "PR lane (recovered suites)" (`.github/workflows/derived-artifacts.yml`), last, after the goal recovery file: its module `pytestmark` is `bootstrap_profile`, so it belongs after the sandboxed suites. `RECOVERED_SUITES_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) pins the same argv. Serially under `--storage-lease-guard --timeout=180`, this file and the goal-dispatch file gave 110 passed and no lease finding in 262.9 s (load 65-84, while 20 other runs went on). `Tests/CI` at `-n 4`: 1240 passed, 1 skipped. Never pr-fast-lane.

**Step timings (AC#3, shared with TASK-39163/39168).** `jobstep.sh fu-r13-load "Run the recovered suites"` (19 targets, serial, `--storage-lease-guard --timeout=180`), 2026-11-10, each 839 passed, 1 skipped, no lease finding:

| Run | pytest time | Load (1 min) |
|---|---|---|
| 1 | 1444.4 s (24.1 min) | 67-78 |
| 2 | 943.3 s (15.7 min) | 36-67 |
| 3 | 713.2 s (11.9 min) | 36-37 |

Run 1 is over the job's 20-min cap, at a load of 67-78 on 18 CPUs (four streams of other agents' runs plus this task's 20 goal-dispatch runs ending as it started). The additions alone, serially under the guard: the goal-dispatch file 94.7 s (load 36-39; it replaces three nodes already in the step) and this library file 143.2 s (load 39-63), about 210 s of run 3's 713 s. CI ran this step in 5.2-5.5 min (job 6.0-6.5 min) on the three latest PR runs that finished it (42841528703, 42841511400, 42737620260, read-only `gh`). Charging the additions at the same share (about 30%) projects about 7.8 min of step and 9 min of job on CI, 11 under the cap; a PR run of this branch is needed to confirm it.

**Files.** `Tests/Chat/test_console_automatic_library_preparation.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
<!-- SECTION:NOTES:END -->
