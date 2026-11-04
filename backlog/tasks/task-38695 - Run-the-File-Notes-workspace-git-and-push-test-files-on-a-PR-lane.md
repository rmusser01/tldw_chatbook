---
id: TASK-38695
title: "Run the File Notes workspace, git and push test files on a PR lane"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 01:52
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

## Implementation Plan

1. The UI census runs entries serially (no xdist) with `--timeout=180`, under a 30 s per-entry budget. A whole file runs for minutes, so each file goes on as one node id per test function. That is the criterion's "every node through split entries".
2. The census has no exclude or deselect syntax; an entry is a file or a node id. The workspace file's four TASK-32185 functions (six red cases) therefore stay out, and its other functions go on.
3. Run each file's entries serially in the lane's pytest configuration 3 times, and record each entry's time (setup, call and teardown, summed over its cases). Split any entry over 30 s by parameter.
4. Append the entries at the census tail, and set `MINIMUM_FILES` to the new count. Run the census checker, the census CI tests, and preflight.

## Implementation Notes

Branch fu/r6-fntests, on top of TASK-38693 and TASK-38694; base c2cdb6b82e. The task stays In Progress: AC#1 and AC#2 are met for the git and push files, but not for the workspace file. See "Not met" below.

- **Lane and entry form.** The files run on the UI Fast Lane, through `scripts/ui_pr_gate_census.txt`. That lane runs serially with `--timeout=180`, and the census holds each entry to 30 s. A whole file runs for minutes, so every test function is its own node id, appended at the tail; no existing slot moves. The census has no exclude or deselect syntax (an entry is a file or a node id). The floor is the new entry count; every new id resolves in `check_ui_pr_gate_census.py`.
- **Entries added:**
  - git: 111 functions, beside the 8 quit and removal nodes that were already listed. That is every test in the file.
  - push: 44 functions, every test in the file.
  - workspace: 109 entries.
- **Workspace changes made to fit the budget.**
  - `test_disabled_file_notes_actions_meet_contrast_in_every_shipped_theme` took 105-123 s in one case. It now runs as eight groups (`[0]`-`[7]`), each checking every eighth of the 72 themes. Together the groups still cover all 72.
  - `test_library_notes_source_choices_render_and_switch_by_keyboard` is listed by size, as `[size1]` and `[size2]`.
- **Measured time,** in serial runs of each file's entries in the lane's pytest configuration (`--timeout=180 --durations=0`, no xdist), at load average 13-22 on 18 CPUs. Totals are setup, call and teardown, summed over cases, at the slowest of three runs:

  | File | Pytest time per run | Largest entry | Median entry |
  |---|---|---|---|
  | git | 204-215 s | 16.3 s (`test_40x20_prepare_scrolls_actions_below_a_long_linked_root`) | 1.9 s |
  | push | 84-85 s | 6.7 s | 1.5 s |
  | workspace | 548-552 s | 22.6 s (`test_file_notes_disclosed_actions_fit_wide_and_compact_layouts`) | 2.9 s |

  The contrast groups took 14.6-17.3 s each. Source choices took 6.1 s at size1 and 7.0 s at size2.
- **AC#2 runs,** 3 of 3 green for each file's entries:
  - git: 158 passed each time;
  - push: 64 passed each time;
  - workspace: 165 passed each time, with the final split.
- **Not verified:** the lane installs a minimal dependency set (`pip install -e . pytest pytest-asyncio pytest-timeout packaging aiohttp-jinja2`) on Linux. These runs used the local macOS venv, so the first CI run of the lane is the check on that.
- **Not met: the workspace file.**
  - Its four TASK-32185 functions stay off the lane, because they are red at base. They are `test_notes_authority_switch_restores_visible_focus_and_typing_owner` (all 3 cases), `test_notes_authority_round_trip_resets_only_transient_work_session`, `test_notes_authority_round_trip_retains_both_workspaces` and `test_wide_files_task_return_restores_database_browse_receipt`. A base whole-file run gave 6 failed, 159 passed.
  - `test_library_notes_source_choices_render_and_switch_by_keyboard[size0]` (40x20) passes, but takes 31-34 s: 31.5 s after a warm-up case, 33.4 s alone. That is over the 30 s per-entry budget; it presses Tab up to 240 times to reach the source buttons.
  - So the workspace file is not "every node" on the lane, and the whole file does not pass 3 of 3. Both need TASK-32185 and a decision on that one slow node.
- Files: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Tests/UI/test_library_file_notes_workspace.py`.

