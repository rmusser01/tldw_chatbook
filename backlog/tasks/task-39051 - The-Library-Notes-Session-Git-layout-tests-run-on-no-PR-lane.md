---
id: TASK-39051
title: "The Library Notes Session Git layout tests run on no PR lane"
status: To Do
created_date: 2026-11-06 10:09
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
- [ ] #1 The file runs whole on a PR lane as a census tail entry, and three timings of that lane's command are recorded (locally about 10 s at `-n 4`)
- [ ] #2 The three receipt nodes pass in that lane's command; if they need the sanctioned `bootstrap_profile` enrolment for that, it is added and shown red (RecoveryRequired) without it
- [ ] #3 `scripts/check_ui_pr_gate_census.py` passes
<!-- AC:END -->
