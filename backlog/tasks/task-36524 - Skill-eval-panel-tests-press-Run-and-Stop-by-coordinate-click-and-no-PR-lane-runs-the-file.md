---
id: TASK-36524
title: Skill-eval panel tests press Run and Stop by coordinate click, and no PR lane runs the file
status: Done
assignee:
- '@claude'
updated_date: '2026-11-03 20:58'
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- flaky
- evals
priority: low
references:
- Tests/UI/test_evals_skill_eval_panel.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-36382
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36382's review round 2 (PR #3227, 2026-10-23). That review moved the file's shared `_pick_via_overlay` helper from a coordinate click to the keyboard, because a click that landed before layout settled left the Select closed.

The file still presses Run and Stop with `pilot.click("#skill-eval-run")` and `pilot.click("#skill-eval-cancel")`, 11 times on dev 5b5920b8e2 (:121-618). Under the reviewer's one-window idle probe (each `pilot.pause()` sees one 20 ms idle window), `test_directory_path_subject_enables_run` failed 3 of 5 runs at its own Run click, both before and after PR #3227.

The file is not in `scripts/ui_pr_gate_census.txt`, on dev or on PR #3227's branch, and no workflow targets it, so these failures reach no PR run. On dev it passes 23 of 23 with `-n 4`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No test in the file presses a button by coordinate; each presses the button directly and waits for the state it asserts
- [x] #2 Each changed test passes 20 of 20 serial single-node runs, and under a one-window idle probe the old click form fails while the new form passes
- [x] #3 The file runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Measure the base: the file with `-n 4`, and `test_directory_path_subject_enables_run` under a one-window idle probe (a scratch `-p` plugin that makes `pilot.pause()`'s idle wait one 20 ms sleep; the file is `bootstrap_profile`, so the plugin reaches the test process).
2. Replace all 11 `pilot.click("#skill-eval-run"/"#skill-eval-cancel")` calls with a `_press` helper: `Button.press()` on the real button, then a bounded wait for the state the test asserts (the RunRequested/CancelRequested record, or the Run guard's toast before asserting nothing was posted). Wait for an Input-driven subject to reach the harness before pressing. Hoist `_pick_via_overlay`'s wait loop so both helpers share it.
3. AC#2: 20 serial single-node runs per changed test; old click form vs new press form under the probe.
4. AC#3: append the file at the census tail (whole file if it fits the 30 s per-file budget, else node ids), set `MINIMUM_FILES` to the entry count; run the census checker and the `Tests/CI` census and shard tests.
5. File `-n 4`, preflight, notes.

## Implementation Notes

Test-only; no product code changed. The file's 11 coordinate presses of Run and Stop now go through `_press(pilot, id, until, what)`. It calls `Button.press()` on the real button, the call a click and Enter both end in, then waits (bounded, 6 s) for the state the test asserts. A "posts nothing" assertion first waits for the Run guard's toast, which proves the press was handled. The directory-path test waits for the typed path's `SubjectChanged` before it presses. `_pick_via_overlay`'s wait loop moved to module level as `_wait_for`, shared by both helpers; `_pick_via_overlay`'s signature is unchanged (`test_evals_skill_eval_screen.py` imports it).

**Cause.** A scratch diagnostic plugin logged every press and click. Under the one-window idle probe, the directory test's second Run click landed on the button and pressed nothing: `Button._on_click` ignores a click while the 0.2 s `-active` class from the previous press is still set (the trap recorded in `lessons-testing-evidence.md`, 2026-08-06). The same log showed `test_run_without_subject_posts_nothing`'s sentinel-subject click dropped the same way while the test still passed; its "nothing posted" check was vacuous. `test_run_without_both_models_picks_posts_nothing` set no subject, so its press hit the subject guard, which runs before the model guard. It now sets a subject and asserts the model guard's toast. With the model guard removed by a temporary source edit (reverted), the old form of that test passed and the new form failed.

**Evidence** (load average 20-31 on 18 CPUs; the probe is a scratch `-p` plugin that makes `pilot.pause()`'s idle wait a single 20 ms sleep, and the file is `bootstrap_profile`, so it runs in the probed process):
- Base, old click form, under the probe: `test_directory_path_subject_enables_run` failed 5 of 5 (`assert 0 == 1` at its final Run click). The other six click tests passed 5 of 5.
- Clicks put back inside the new helper (temporary edit), under the probe: the two multi-press tests failed 5 of 5 each (`subject guard: condition never became true`, `RunRequested: condition never became true`). The five single-press tests passed 5 of 5, so this probe does not separate the two forms for a single press.
- New form under the probe: 70 of 70 (10 per changed test).
- AC#2: 20 of 20 serial single-node runs for each of the 7 changed tests (140 of 140).
- The file with `-n 4`: 23 passed. `test_evals_skill_eval_screen.py` with `-n 4`: 32 passed.
- AC#3: the file is appended whole at the census tail. Serially, as the lane runs it, it takes 9-11 s pytest (11-14 s wall) at load average ~30, under the 30 s per-file budget. `MINIMUM_FILES` is the entry count. Census checker OK; the `Tests/CI` census, shard and workflow tests passed (423); preflight passed. Ruff lint count unchanged from base; the file was already not `ruff format`-clean at base and was not reformatted.

Files: `Tests/UI/test_evals_skill_eval_panel.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
