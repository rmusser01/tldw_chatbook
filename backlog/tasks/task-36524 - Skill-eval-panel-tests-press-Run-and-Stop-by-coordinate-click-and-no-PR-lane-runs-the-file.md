---
id: TASK-36524
title: Skill-eval panel tests press Run and Stop by coordinate click, and no PR lane runs the file
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- flaky
- evals
priority: low
references:
- Tests/UI/test_evals_skill_eval_panel.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-36382
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36382's review round 2 (PR #3227, 2026-10-23). That review moved the file's shared `_pick_via_overlay` helper from a coordinate click to the keyboard, because a click that landed before layout settled left the Select closed.

The file still presses Run and Stop with `pilot.click("#skill-eval-run")` and `pilot.click("#skill-eval-cancel")`, 11 times on dev 5b5920b8e2 (:144-641). Under the reviewer's one-window idle probe (each `pilot.pause()` sees one 20 ms idle window), `test_directory_path_subject_enables_run` failed 3 of 5 runs at its own Run click, both before and after PR #3227.

The file is not in `scripts/ui_pr_gate_census.txt`, on dev or on PR #3227's branch, and no workflow targets it, so these failures reach no PR run. On dev it passes 23 of 23 with `-n 4`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No test in the file presses a button by coordinate; each presses the button directly and waits for the state it asserts
- [ ] #2 Each changed test passes 20 of 20 serial single-node runs, and under a one-window idle probe the old click form fails while the new form passes
- [ ] #3 The file runs on a PR lane
<!-- AC:END -->
