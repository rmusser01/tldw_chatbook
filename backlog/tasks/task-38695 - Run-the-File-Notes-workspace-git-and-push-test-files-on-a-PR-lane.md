---
id: TASK-38695
title: "Run the File Notes workspace, git and push test files on a PR lane"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- tests
- library
- ci
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_git.py
- Tests/UI/test_library_file_notes_git_push.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-38693
- TASK-38694
- TASK-32185
- TASK-36513
- TASK-36528
- TASK-36531
- TASK-38646
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down. Checked on dev 9bbd5c2515, 2026-11-03.

No PR lane runs most of the three File Notes UI test files:
- `Tests/UI/test_library_file_notes_workspace.py`: 103 test functions;
- `Tests/UI/test_library_file_notes_git.py`: 110;
- `Tests/UI/test_library_file_notes_git_push.py`: 44.

None of them is in `scripts/ui_pr_gate_census.txt` or a workflow target. PR #3377 adds 8 entries (65 cases) for the git file's new quit and removal tests only. Nightly Deep last ran on 2026-09-27.

Their reds on dev (2026-11-03, whole-file runs with `-n 4`) are filed separately, and this lane should follow them:
- **workspace**: the same 10 nodes failed in each of two runs (152 passed). These are covered by TASK-32185 (the authority-switch, round-trip and Files task-return tests), TASK-36513 (the linked-root identity row), TASK-36528 (the Save failed contrast and the Offline elision) and TASK-36531 (the database files switch). TASK-38646 covers a further flake.
- **git**: the two 40x20 reds and two flakes in TASK-38693.
- **push**: the twelve reds in TASK-38694.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three files run on a PR lane (whole-file entries, or every node through split entries), with their measured time recorded in the task notes
- [ ] #2 Each file passes 3 of 3 runs in that lane's configuration before it is added
<!-- AC:END -->
