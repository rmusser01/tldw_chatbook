---
id: TASK-42281
title: Lane the prompt-hook and fresh-profile startup regressions in the PR fast lane
status: Done
assignee:
  - '@claude'
created_date: '2026-11-22 01:20'
labels:
  - ci
  - lanes
priority: medium
dependencies:
  - TASK-42172
  - TASK-42173
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-42172 (a fresh profile with no config folder was refused at startup on Linux) and TASK-42173 (UserPromptSubmit hooks never ran on Console sends) each added a regression test. Both tests sit in files that run in no PR lane; only the nightly full suite runs them. So a regression would reach dev and be caught a day late. Owner decision (2026-11-22): put both in the required PR fast lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fresh-profile startup regression runs in a required PR fast lane
- [x] #2 The UserPromptSubmit hook regression runs in a required PR fast lane
- [x] #3 The CI queue-pressure contract pins both additions and passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the startup node to the sandboxed PR Fast Lane (not bootstrap_profile) and the hook node to admission-sensitive lane 4 (bootstrap_profile, beside the other Console agent-turn files), in derived-artifacts.yml.
2. Mirror both in FAST_LANE_TARGETS and ADMISSION_4_TARGETS in Tests/CI/test_ci_queue_pressure_contract.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Lanes:** each test is added as a node ID, not its whole file, to keep the cost small.
  - `test_fresh_profile_without_a_config_folder_starts_normally` goes in the PR Fast Lane: three cases, under 1 s.
  - `test_prompt_hook_runs_while_the_submit_task_holds_its_agent_admission` goes in PR Fast Lane (admission-sensitive 4): two cases, about 1.7 s. It is a `bootstrap_profile` module and needs native POSIX group custody.
- **Contract:** both target tuples carry the nodes with comments. `Tests/CI` passes: 1811 passed, 2 skipped.
- **Checks:** each node passes when run the way its lane runs it (`--timeout=180` and `--timeout=300`).
- **Stacking:** stacked on PR #3683, which adds both tests. The lane PR opens after #3683 lands, so a missing node can never turn a lane red.
<!-- SECTION:NOTES:END -->
