---
id: TASK-34986
title: The Console P0 regression gate runs close to its 20-minute cap
status: In Progress
created_date: 2026-10-14 16:13
labels:
- ci
- tests
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-14 19:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console P0 regression gate is one of the jobs behind the required check, with a 20-minute cap. Over the PR runs measured on 2026-10-14 its median was 13.0 minutes and its slowest 18.7; it took 16 minutes 30 seconds on PR #3106's first run. When it reaches its cap GitHub cancels it and leaves no test report, and the merge queue counts that against the PR. TASK-34916 and TASK-34917 made room in the PR Fast Lane and the UI shards but did not cover this gate. The owner's rule for CI time is to add a shard rather than raise the minutes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The gate's slowest run over at least 20 pull-request runs leaves at least 4 minutes under its cap
- [x] #2 No test is dropped from the required check to achieve it
- [x] #3 The contract tests that pin the gate reflect the new layout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the gate per job, per step and per file from recent pull_request runs (gh api jobs + job logs), including runs after #3106 merged.
2. The gate holds two Console P0 steps (sandboxed, admission-sensitive) and PR2921's two Dreams/Guardian steps; replay the measured runs against splits and pick the one that leaves >= 4 min under the 20-min cap on the slowest run without growing an existing job.
3. Move the two Dreams/Guardian steps, verbatim, into their own job with the same install line and the 20-min cap; add it to the required aggregate's needs with a verdict step.
4. Update the contract tests that pin the gate (test_ci_queue_pressure_contract, test_derived_artifacts_workflow, test_console_p0_regression_gate, test_response_rules_critical_gate) to pin the new layout exactly, and show mutants red.
5. ADR-103 note; record queue delay and the required check's wall clock; Tests/CI, YAML parse, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Split the Console P0 regression gate along its two owners instead of raising the cap: PR2921's two Dreams/Guardian steps moved, verbatim, into a new job dreams-boundary ('Dreams and Guardian boundary contracts').

Measured (gh api jobs for 83 pull_request runs of derived-artifacts.yml, 2026-10-13 01:53Z to 2026-10-14 19:35Z; 76 with every step completed): the gate took median 16.1 / p90 17.4 / max 18.7 min, and 2 runs were cancelled at the 20-min cap (38493363303, 38500380448, 20.3 min). The 13 runs on the post-#3106 layout: median 16.3 / max 18.0. (The 13.0 median in the task description predates the Dreams steps landing in this job with #2921.) Per step over the 76 runs, median / max: sandboxed Console P0 2.94 / 3.48, admission-sensitive Console P0 7.79 / 9.65, Dreams boundary 1.83 / 2.05, Dreams admission-sensitive 2.57 / 3.90, setup ~0.8. Per file (12 green runs' logs, progress-line timestamps): the admission step is dominated by Tests/UI/test_console_row_menu_action_sweep.py (median 117 s), test_console_hook_review_send_freeze (50 s), test_response_rules_write_contention (49 s), test_deepseek_explicit_mode (37 s); the sandboxed step by test_app_keep_alive_dead_screen (46 s) and test_console_conversation_action_menu (43 s); the Dreams admission step by test_watchlists_bulk_source_authoring (123 s).

Choice (replay of the same 76 runs: each half = the job's own setup + the steps it keeps): console-p0-gate with the two Console steps median 11.6 / p90 12.7 / max 13.9 min (6.1 min under the cap; AC#1 asks 4); dreams-boundary median 5.2 / p90 5.7 / max 6.7. A balanced split (admission alone 10.4 max, sandboxed + Dreams 9.8 max) was rejected: it splits the Console P0 sandboxed/admission pair across jobs and mixes owners for 3.5 min. Not into dreams-tracking: that job varies 6.2-12.7 min (250 tests since #3108; one run 685 s of tests), and these steps would take it to ~18.6.

Change: both steps keep their targets, order, --timeout (180 / 300) and 'if: ${{ !cancelled() }}' (parsed comparison of every pytest step before/after: argv and if identical, only the job changed); same install line, 20-min cap (the one they had). Each step is still its own pytest process, the admission-sensitive one second (TASK-32873). The required aggregate needs dreams-boundary and has a 'Require successful Dreams and Guardian boundary contracts' verdict. console-p0-gate keeps 5 steps. One more runner per PR (ADR-103 note): eleven test runners.

Contracts: Tests/CI/test_ci_queue_pressure_contract.py pins dreams-boundary exactly (name, if, runs-on, 20-min cap, 5 steps, install line, the two steps last in order, exact argv with --timeout, sole owner) and console-p0-gate at 5 steps with its two Console steps last; a 'moved-back' mutation joins the placement-drift set. test_derived_artifacts_workflow.py (job order, verdict table = needs, PR-lane step locations incl. moved-back), test_console_p0_regression_gate.py (lane table), test_response_rules_critical_gate.py (STEP_JOBS: each inherited step's job, both jobs 5 steps / 20 min / in needs). Mutants applied to the real workflow, each red in Tests/CI, file restored byte-identical: Dreams admission step moved back into console-p0-gate (112 failed), dreams-boundary dropped from needs (3 failed), its cap 20 -> 25 (2 failed), Tests/Guardian/test_crisis_resources.py dropped from the boundary step (2 failed), its Dreams admission --timeout=300 -> 30 (2 failed), its verdict step removed (12 failed).

Queue delay and wall clock (current layout, the 11 runs since #3106 merged with every lane finished): lane jobs start median 2.0 / p90 21.9 / max 24.8 min after the run starts (3.8 / 22.0 / 40.2 from run creation, which includes merge-queue approval holds up to 35.9 min); required check wall clock (run created -> aggregate done) median 28.9 / p90 43.6 / max 57.7 min, against the pre-#3106 baseline of 39.0 / 66.8 / 256.1 (81 runs). This change adds one runner per PR; the PR's own run and the 20-run measurement will show whether queueing eats the gain.

AC#1 left unchecked: it needs 20 measured runs of the new layout. Verify with the gh api jobs query: max 'Console P0 regression gate' <= 16 min.

Tests: Tests/CI + Tests/Packaging/test_python_runtime_floor.py 930 passed, 1 skipped; YAML parses; ruff findings on the four touched tests identical per file and message to HEAD.

Files: .github/workflows/derived-artifacts.yml, Tests/CI/test_ci_queue_pressure_contract.py, Tests/CI/test_derived_artifacts_workflow.py, Tests/CI/test_console_p0_regression_gate.py, Tests/CI/test_response_rules_critical_gate.py, backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
