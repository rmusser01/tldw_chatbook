---
id: TASK-27018
title: Three Console command-composer tests are red on dev
status: To Do
assignee: []
created_date: '2026-09-01 19:04'
labels:
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deterministic on dev at 3f30fb686 (whole file, -p no:randomly, 3 failed / 100 passed, identical on pristine dev and feature branches): test_raw_cli_collapsed_state_retains_danger_label_and_one_row_geometry, test_console_unknown_command_second_unmodified_enter_sends_as_text, test_console_collapsed_paste_starting_with_slash_sends_normally. Recorded on TASK-25715's ledger as finding 6; this task gives the trio an owner. Bisect not yet run -- start there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All three tests pass, or each is individually re-decided against current composer behaviour
- [ ] #2 The cause is bisected to a commit before any fix is written
<!-- AC:END -->
