---
id: TASK-38722
title: "The Console rail-prune tests run on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: '2026-11-05 22:26'
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

**Placement.** `Tests/Chat/test_console_rail_state_prune.py` is appended as a whole-file target to the sandboxed fast PR contract step (`pr-fast-lane`, "Run fast PR contract" in `derived-artifacts.yml`), right after TASK-38747's census rider. It is also appended to `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`. The file is pure: it has no `bootstrap_profile` mark and mounts no app (its dispatch tests drive `ChatScreen`'s prune methods on a stand-in). So it belongs in the sandboxed step, not an admission-sensitive job, whose targets must stay disjoint from it.

**Measured (AC#2, before adding it).** Run 3 times with that step's flags, `pytest -p no:xdist -p no:cov Tests/Chat/test_console_rail_state_prune.py --timeout=180 --tb=short`, through the scratch wrapper, at load average 33-35. The lane's install set has neither plugin, hence the two `-p no:` flags. Results: 3 of 3 passed, 9 tests each, 2.74 s, 2.74 s and 3.34 s in pytest (7.3-8.4 s wall). The step's last four targets (the two TASK-38702 censuses, the quit census and this file) also passed together in one serial process: 199 passed in 60 s.

**Checks.** `Tests/CI` (`-n 4`): 1211 passed, 1 skipped. It includes `test_fast_lane_target_set_is_exact_and_non_overlapping` and the disjoint-target and node-resolution checks in `test_derived_artifacts_workflow.py`. Preflight passed; the size ratchets passed (256).

**Files.** `.github/workflows/derived-artifacts.yml` (one line), `Tests/CI/test_ci_queue_pressure_contract.py`.

