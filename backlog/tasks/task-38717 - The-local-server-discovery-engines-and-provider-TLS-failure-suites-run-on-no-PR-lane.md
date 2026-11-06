---
id: TASK-38717
title: "The local-server discovery engines and provider TLS failure suites run on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: '2026-11-05 21:27'
labels:
- follow-up
- ci
- tests
priority: medium
references:
- Tests/Chat/test_local_server_discovery_engines.py
- Tests/Chat/test_provider_tls_failures.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-36532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36532). Checked on dev c2cdb6b82e, 2026-11-03.

TASK-36532 put `Tests/Chat/test_local_server_discovery.py` on the "Run Dreams and Guardian boundary contracts" step of the `dreams-boundary` job (`derived-artifacts.yml:1090-1116`). Two sibling suites from the same provider-setup work (TASK-34100.6) still run on no PR lane:
- `Tests/Chat/test_local_server_discovery_engines.py` (10 test functions);
- `Tests/Chat/test_provider_tls_failures.py` (10 test functions).

Neither is in `scripts/ui_pr_gate_census.txt`, which takes only `Tests/UI`, or a target in any workflow. Only `test.yml`, which runs on a push to main or on dispatch, and the scheduled `nightly-deep.yml` collect them, as part of the whole `Tests/` directory. Both passed at the base in one `-n 4` run alongside other files.

A Tests/Chat file needs a workflow lane target. The owner allowed such a target in `derived-artifacts.yml` for TASK-36532.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both files run on a PR lane as whole-file targets, with their measured time in that lane recorded in the task notes
- [x] #2 Each file passes 3 of 3 runs in that lane's configuration before it is added
<!-- AC:END -->

## Implementation Plan

1. Lane: the sandboxed "Run Dreams and Guardian boundary contracts" step of the `dreams-boundary` job, which TASK-36532 gave the sibling `Tests/Chat/test_local_server_discovery.py`. Both files are sandboxed (`local_server_probe` mark, no `bootstrap_profile`), and pr-fast-lane is still at its cap.
2. Before adding: run each file 3 times with that step's command (`pytest <file> --timeout=180 --tb=short`, serial, through run.sh) at the base and record each run's time. A file red at the base is not added.
3. Append both files after `Tests/Chat/test_local_server_discovery.py` in that step, and extend the step's TASK-36532 comment. Keep the diff to those lines, because owner draft #3342 also edits the file.
4. Update the lane-contract pins in `Tests/CI` (`DREAMS_BOUNDARY_TARGETS` in `test_ci_queue_pressure_contract.py`, the step list in `test_response_rules_critical_gate.py`) and run `Tests/CI`.

## Implementation Notes

Both files now run whole in the sandboxed "Run Dreams and Guardian boundary contracts" step of the `dreams-boundary` job, right after their sibling `Tests/Chat/test_local_server_discovery.py` (TASK-36532).
- Lane choice: both files are sandboxed (`local_server_probe`, no `bootstrap_profile` node), so they belong in that step, not the admission-sensitive one. pr-fast-lane is still at its cap, which is why TASK-36532 chose this job.
- `.github/workflows/derived-artifacts.yml`: two target lines and nothing else, because owner draft #3342 also edits this file. The reason and the timings are in the `Tests/CI` pin's comment.
- Lane pins updated: `DREAMS_BOUNDARY_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`, and the step's argv in `Tests/CI/test_response_rules_critical_gate.py`. With the workflow reverted and the pins updated, `test_dreams_critical_boundaries_are_executable_required_lane_members` and `test_rule_critical_targets_execute_without_altering_inherited_processes` both failed. With both updated, `Tests/CI` gave 1199 passed and 1 skipped (`-n 4`).
- AC#2, measured before the targets were added. Each file ran with the step's command (`pytest <file> --timeout=180 --tb=short`, serial, through run.sh) at this branch's head, after TASK-36516 and TASK-36517. Load average 43-59 on 18 CPUs:

  | File | Run 1 | Run 2 | Run 3 |
  | --- | --- | --- | --- |
  | `test_local_server_discovery_engines.py` (10 tests) | 10 passed, 9.13 s | 10 passed, 8.31 s | 10 passed, 9.26 s |
  | `test_provider_tls_failures.py` (13 tests) | 13 passed, 45.66 s | 13 passed, 43.23 s | 13 passed, 42.81 s |

  pytest-reported times; each process took 18-24 s and 51-59 s wall, with import and collection. One run of each with the base (29592ecc9b) source was also green: 10 passed in 9.36 s and 13 passed in 46.13 s. The three Chat files together in one process, as the step runs them: 115 passed in 78.93 s.
- About 33 s of the TLS file is one test, `test_real_tls_stalls_are_timeouts_not_plain_http[handshake-stall]`. Its `_never_answer` handler sleeps 30 s after its read, and `async with server` waits for that connection when it closes. Listed as a proposed follow-up; not changed here.

Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`.

