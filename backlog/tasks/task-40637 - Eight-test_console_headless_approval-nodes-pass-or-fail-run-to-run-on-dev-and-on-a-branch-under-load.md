---
id: TASK-40637
title: "Eight test_console_headless_approval.py nodes pass or fail run to run, on dev and on a branch, under load"
status: To Do
assignee: []
created_date: '2026-11-12 15:21'
labels:
  - follow-up
  - tests
  - console
  - flaky
dependencies: []
priority: low
references:
  - Tests/UI/test_console_headless_approval.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviving PR #3446 (TASK-33620.15.53) on 2026-11-12. Other sessions' pytest runs kept the host heavily loaded throughout. The PR does not touch the approval code. On origin/dev 4b90eb860b and on the rebased branch alike, these nodes in `Tests/UI/test_console_headless_approval.py` failed in some runs and passed in others. Evidence: whole file 2x per side, run in parallel; node-only reruns 2-3x per side; one node at a time 3x per side.

- `test_background_decision_updates_runtime_attention_until_resolution`: alone, dev passed 1/3 and the branch 2/3. It passed in all four whole-file runs. `runtime.console_needs_attention` is still False just after `_settle` sees the armed round (`:1151`).
- `test_each_hidden_decision_id_emits_one_sanitized_notice`: alone, 2/3 on each side, with two notices of three (`:1186`).
- `test_hidden_announcer_rejects_session_type_and_id_mismatch`: alone, dev 2/3 and the branch 3/3. Whole file: dev 1/2, the branch 2/2.
- `test_two_rapid_background_rounds_announce_exact_ids_without_legacy_park`: whole file, dev 1/2 and the branch 0/2. Node-only, 1/2 on each side.
- `test_reprojection_never_renders_an_exhausted_head[over-approval]`: whole file, dev 1/2 and the branch 1/2.
- `test_reprojection_never_renders_an_exhausted_head[exact-skill_script]`: whole file, dev 1/2 and the branch 2/2.
- `test_reprojection_never_renders_an_exhausted_head[over-skill_script]`: red once, in a whole-file run on the branch. Green in five node-only reruns, three of them on dev.
- `test_mixed_background_rounds_announce_individually_and_cleanup_exactly`: whole file, dev 2/2 and the branch 1/2.

The first two share a shape, and the others may too. The test waits until a worker thread's approval round shows up in `_pending_approval_rounds`, then asserts state that the same thread publishes only afterwards: the attention projection, or the third notice. Under load the assertion can run between the two. The file runs on no PR lane. Eight other nodes in it fail in every run on both sides; they are not in this task's scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the eight nodes passes 20 of 20 consecutive runs, alone and in file order, on a loaded host (for example beside a `-n 8` run of another suite)
- [ ] #2 Each fix waits on the state it asserts rather than on an earlier signal from the same worker thread, and adds no fixed sleep and no wider timeout
<!-- AC:END -->
