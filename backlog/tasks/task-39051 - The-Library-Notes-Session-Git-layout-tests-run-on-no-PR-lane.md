---
id: TASK-39051
title: "The Library Notes Session Git layout tests run on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 16:13
labels:
- follow-up
- ci
- library
- tests
priority: low
references:
- Tests/UI/test_library_notes_w5_session_git_layout.py
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- Tests/conftest.py
dependencies:
- TASK-38984
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the round-10 Library branch that lands TASK-38984 (not yet a final pushed head). TASK-38984's notes say "The file still runs on no PR lane; the criteria do not ask for one".

- `Tests/UI/test_library_notes_w5_session_git_layout.py` (10 nodes) is in no `pull_request` job: `grep -n "w5_session_git" .github/workflows/*.yml scripts/ui_pr_gate_census.txt` finds nothing on dev or on the Library branch. Only `test.yml` (push to main) and the scheduled nightly run reach it. TASK-32615 wrote the file without a lane AC.
- Three runs at `-n 4` (load 16.5-21.5):
  - On dev: 4 passed, 6 failed, in 8.7-10.1 s. The failures are:
    - the three `test_the_trust_dialog_never_breaks_the_path_it_asks_consent_for[size0-2]` reds TASK-38984 fixes (`assert '           /...4/power/vault' == '/Users/macbo...4/power/vault'`);
    - the three `test_an_action_receipt_never_paints_under_the_danger_heading[size0-2]` nodes, which fail locally with `RecoveryRequired: raw_source_selection_changed` (TASK-38111).
  - After TASK-38984: 7 passed, 3 failed (the same three RecoveryRequired nodes), in 9.2-9.8 s.
- Under a temporary `bootstrap_profile` marker, the three receipt nodes passed 3 of 3, on dev and after TASK-38984. So the file is green apart from the local trap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file runs whole on a PR lane as a census tail entry, and three timings of that lane's command are recorded (locally about 10 s at `-n 4`)
- [x] #2 The three receipt nodes pass in that lane's command; if they need the sanctioned `bootstrap_profile` enrolment for that, it is added and shown red (RecoveryRequired) without it
- [x] #3 `scripts/check_ui_pr_gate_census.py` passes
<!-- AC:END -->

## Implementation Plan

1. Read recent UI Fast Lane shard times read-only, and time the whole file with the lane's own command (about 10 s at `-n 4` locally), so it fits a census entry near 30 s.
2. Check whether the three receipt nodes pass in that command without a `bootstrap_profile` enrolment; add the sanctioned enrolment only if they need it, shown red (`RecoveryRequired`) without it.
3. Append the whole file at the census tail, raise `MINIMUM_FILES` to the measured entry count, and run `scripts/check_ui_pr_gate_census.py`; record three timings of the lane command.

## Implementation Notes

- **Lane.** `Tests/UI/test_library_notes_w5_session_git_layout.py` is appended whole at the UI census tail, and `MINIMUM_FILES` is raised to the measured entry count. Recent UI Fast Lane shards (run 42218508176, #3436, 2026-11-06) took 558-888 s of their 20-minute cap; this file adds about 20 s to one shard.
- **Timings (AC#1).** These used the lane's own command: serial, since the lane installs no xdist, with `--timeout=180 --durations=0 --durations-min=0 --tb=short`, through run.sh. 10 passed each time, in 15.6, 17.5 and 20.5 s of pytest time (18.3, 20.4 and 23.8 s wall), at load 11.7-12.2 on 18 CPUs.
- **Receipt nodes (AC#2).** Without an enrolment, the lane's command fails the three `test_an_action_receipt_never_paints_under_the_danger_heading[size0-2]` nodes with `RecoveryRequired: raw_source_selection_changed` (3 failed, 7 passed). This is the only test in the file that mounts the production TldwCli (`_production_workspace_context` -> `_build_test_app` -> `load_settings`). It now carries TASK-32873's per-node `@pytest.mark.bootstrap_profile`, which the sibling File Notes suites get from conftest's list. With the marker, all 10 nodes pass, three times out of three.
- **AC#3.** `scripts/check_ui_pr_gate_census.py`: "OK: 1749 Tests/UI entries in the PR gate (floor 1749)". `Tests/CI` at `-n 4`: 1215 passed, 1 skipped.
- Files: `Tests/UI/test_library_notes_w5_session_git_layout.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
