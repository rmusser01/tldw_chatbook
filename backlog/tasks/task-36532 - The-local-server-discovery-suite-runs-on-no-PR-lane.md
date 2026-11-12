---
id: TASK-36532
title: The local server discovery suite runs on no PR lane
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:44
updated_date: '2026-11-03 21:23'
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/Chat/test_local_server_discovery.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-36381
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36381's review (branch fu/h-settingsreds, 2026-10-23), checked on dev 5b5920b8e2. No PR lane runs `Tests/Chat/test_local_server_discovery.py` (59 tests):
- the UI census takes only Tests/UI;
- `derived-artifacts.yml` has a lane with a Tests/Chat target list, and this file is not on it.

TASK-36381 fixes the file's one red node (`test_probe_handles_recursive_json_as_bounded_failure`). It leaves the lane, because that burn-down could not edit the workflow. Draft PR #3036 also changes `derived-artifacts.yml`, so a PR for this task may be evicted from the merge queue for a workflow-file rebase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tests/Chat/test_local_server_discovery.py runs on a PR-required lane, with a matching Tests/CI lane-contract pin where that lane has one
- [x] #2 The lane's measured job time stays within its timeout, with the added time recorded in the task notes
<!-- AC:END -->

## Implementation Plan

1. Append `Tests/Chat/test_local_server_discovery.py` as the last rider of `pr-fast-lane`'s sandboxed "Run fast PR contract" step in `.github/workflows/derived-artifacts.yml` (the PR lane that already runs the sandboxed `Tests/Chat` contract riders; the file is pure `httpx.MockTransport` tests, not `bootstrap_profile`). One added line, away from the `runs-on` lines owner draft #3342 changes.
2. Pin it at the tail of `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`, with a comment naming this task, so dropping it from the step fails the exact-target contract.
3. Measure the file's serial runtime in the lane's install set and read recent `PR Fast Lane` job times on CI; record the added time against the 35-min cap.
4. Run `Tests/CI -n 4`, the discovery file with `-n 4`, and preflight.

## Implementation Notes

The suite now runs whole in `dreams-boundary`'s sandboxed step "Run Dreams and Guardian boundary contracts", the last target before `--timeout=180`. That is a one-line change to `.github/workflows/derived-artifacts.yml`, nowhere near the `runs-on` lines owner draft #3342 edits.

**Deviation from plan step 1 (the lane).** The plan named `pr-fast-lane`. The measured job times rule it out:
- **`pr-fast-lane`:** the GitHub API job times for its 37 successful PR runs on 2026-11-03 were median 31.4 / max 34.7 min (pytest step median 30.5 / max 33.9) against the 35-min cap. Its comment still gives the 2026-10-14 figures, median 16.1 / max 20.8. With this file added, the worst case would reach the cap.
- **Other sandboxed steps:** `console-p0-gate` ran median 18.3 / max 20.0 of its 20 min. `dreams-tracking` pins one target. The UI census takes only `Tests/UI`.
- **Admission jobs:** every target in them is `bootstrap_profile`, and this file is a sandboxed suite.

`dreams-boundary` had headroom: 51 successful runs at median 6.2 / max 7.2 min against its 20-min cap, with this step at median 1.82 / max 2.20 min. The step already carries non-Dreams riders (Watchlists, Notifications). No new job was added.

**Pins (AC#1).** The step's argv is pinned exactly in two places, and both now include the file at the tail:
- `DREAMS_BOUNDARY_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`, with a comment giving the reason;
- the `INHERITED` table of `Tests/CI/test_response_rules_critical_gate.py`.

**Cost (AC#2).** The file alone, serial and invoked as the lane runs it (`--timeout=180 --tb=short`), took 12.8-15.2 s over four runs at load 27-31 on 18 CPUs: 59 passed, 12.4 s in setup, call and teardown. The whole step serial took 221 s with the file (223 passed) and 252 s without it (164 passed) at load 33-43. Load noise is larger than the file's cost there. Expected added CI time is about 0.2-0.4 min, which puts the job's max at about 7.5 of its 20 min.

**Red/green.** With both pins in place and the workflow line removed, `test_dreams_critical_boundaries_are_executable_required_lane_members` and `test_rule_critical_targets_execute_without_altering_inherited_processes` both failed on their argv assertion. With the line restored, both passed. The reverse check also failed: with the workflow line present and the `INHERITED` pin missing, the run failed with "At index 19 diff: 'Tests/Chat/test_local_server_discovery.py' != '--timeout=180'".

**Runs (`-n 4`, load 38-55):**
- `Tests/CI`: 1177 passed, 1 skipped.
- `Tests/Chat/test_local_server_discovery.py`: 59 passed.
- `./scripts/preflight.sh`: passes.

**Files:**
- `.github/workflows/derived-artifacts.yml`
- `Tests/CI/test_ci_queue_pressure_contract.py`
- `Tests/CI/test_response_rules_critical_gate.py`
- `backlog/docs/lessons-testing-evidence.md`: the `dreams-boundary` double pin, and a new entry on stale lane-timing comments.
