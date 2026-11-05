---
id: TASK-38722
title: "The Console rail-prune tests run on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: '2026-11-05 23:12'
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/Chat/test_console_rail_state_prune.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-36518
- TASK-36532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03.

`Tests/Chat/test_console_rail_state_prune.py` has 9 test functions. They pin `_prune_console_rail_preferences`, the launch-time prune of Console rail preferences that TASK-36518 reworked. No PR lane runs the file:
- it is not in `scripts/ui_pr_gate_census.txt`, which takes only `Tests/UI`;
- it is not a target in any workflow.

Only `test.yml` (on a push to main or on dispatch) and the scheduled `nightly-deep.yml` collect it, with the whole `Tests/` tree. It passed at the base in one `-n 4` run through the scratch-profile wrapper.

A Tests/Chat file needs a workflow lane target. The owner allowed one in `derived-artifacts.yml` for TASK-36532 (`dreams-boundary`, :1092-1116).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file runs on a PR lane as a whole-file target, with its measured time in that lane recorded in the task notes
- [x] #2 It passes 3 of 3 runs in that lane's configuration before it is added
<!-- AC:END -->

## Implementation Plan

1. Pick the lane: the file is pure (no `bootstrap_profile`, no app mount), so it joins the sandboxed fast PR contract step (`pr-fast-lane`, "Run fast PR contract"), whose last riders are TASK-38702's whole-file Tests/Chat censuses.
2. Run it 3 times with that step's flags (`pytest <file> --timeout=180 --tb=short`, serial) before adding it, and record the times.
3. Append it as a whole-file target in `derived-artifacts.yml`, next to TASK-38747's census rider, and to `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`; run the `Tests/CI` lane-contract tests.

## Implementation Notes

**Placement.** `Tests/Chat/test_console_rail_state_prune.py` is a whole-file target at the end of the "Run Dreams and Guardian boundary contracts" step (`dreams-boundary` in `derived-artifacts.yml`), right after TASK-38747's quit-census rider. It is pinned in `DREAMS_BOUNDARY_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) and in that step's argv pin in `Tests/CI/test_response_rules_critical_gate.py`. The file is pure: it has no `bootstrap_profile` mark and mounts no app (its dispatch tests drive `ChatScreen`'s prune methods on a stand-in). So it belongs in a sandboxed step, not an admission-sensitive one. It was first appended to `pr-fast-lane` and moved on review r8g-rev1. That job ran median 32.0 and max 34.8 min of its 35 over 54 runs (2026-11-04/05), and 8 runs were cancelled at the cap. `dreams-boundary` runs median 6.7 and max 7.8 of its 20, as with TASK-36532's rider.

**Measured (AC#2).** Both steps use the same install line (no xdist, no coverage plugin) and the same flags. Before it was added, 3 runs with those flags (`pytest -p no:xdist -p no:cov Tests/Chat/test_console_rail_state_prune.py --timeout=180 --tb=short`, through the scratch wrapper, load average 33-35): 3 of 3 passed, 9 tests each, 2.74 s, 2.74 s and 3.34 s in pytest. After the move, 3 more runs with the step's command over the two moved targets (load average 15-16): 3 of 3 passed, 15 tests each, 4.55 s, 3.81 s and 3.07 s in pytest. The census scan took 1.78-2.58 s of that, so this file took about 1-2 s. The whole step, run once locally with both riders in one process: 238 passed in 141 s.

**Checks.** `Tests/CI` (`-n 4`): 1211 passed, 1 skipped. That covers both step pins and the disjoint-target and node-resolution checks in `test_derived_artifacts_workflow.py`. Preflight passed; the size ratchets passed (256).

**Files.** `.github/workflows/derived-artifacts.yml` (one line), `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`.
