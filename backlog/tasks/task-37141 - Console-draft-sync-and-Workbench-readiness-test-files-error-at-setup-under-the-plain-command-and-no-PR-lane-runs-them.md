---
id: TASK-37141
title: >-
  Console draft-sync and Workbench readiness test files error at setup under
  the plain command and no PR lane runs them
status: To Do
assignee: []
created_date: '2026-10-27 18:45'
labels:
  - console
  - tests
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while gating TASK-33620.15.22 (2026-10-27, compared by test id against a detached origin/dev 416feb50ee worktree). The test files that pin how a composer draft edit reaches Send's and the Workbench's readiness never reach their assertions under the plain command on a developer machine: every case errors at setup with `RecoveryRequired: raw_source_selection_changed`, identically on dev and on the branch. None of them is in scripts/ui_pr_gate_census.txt, so CI does not run them on a PR either. Verified here:

- `Tests/UI/test_console_composer_draft_changed.py`: 23 of 23 error at setup.
- `Tests/UI/test_console_draft_sync_equality_gate.py`: 9 of 9 error at setup. With a scratch plugin that marks every item `bootstrap_profile`, 8 pass and 1 fails (`test_a_genuinely_new_workbench_state_still_repaints_every_static`) on both trees.
- `Tests/UI/test_console_run_status_wiring.py`: 1 of 1 errors at setup. With the marker it fails on both trees (`_console_run_active()` is False).

Some cases of further composer and send files fail with the same error while the rest of the file runs: on dev 416feb50ee, `test_console_composer_collapse.py` passed 39 cases before one failed with it, and `test_console_command_composer.py` passed 43 (both run with -x). The same gate's earlier comparison on dev 3be493413a also saw it in `test_console_composer_improvement_transaction.py` and `test_console_pending_attachment_stash.py`. This is the TASK-36014 class (that task lists the avatar, rail and voice-chip suites); these files are not on its list. The cost is concrete: TASK-33620.15.22 changed the Workbench's Send readiness, which these files pin, and only a hand-run with the scratch plugin showed it had not regressed them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each file named here runs to its assertions under the plain command on a developer machine
- [ ] #2 Each file named here that passes on dev runs on a PR lane, or the task records why it cannot
- [ ] #3 Reds that remain once the files run are fixed or filed by test id
<!-- AC:END -->
