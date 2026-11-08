---
id: TASK-39152
title: "Re-pin the 86 Library test nodes in test_library_shell.py and test_library_ingest_flow.py that fail on their own assertions once TASK-39141 lifted the RecoveryRequired wall"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_shell.py
- Tests/integration/test_library_ingest_flow.py
- Tests/conftest.py
dependencies:
- TASK-39141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39141 on round-12 branch fu/r12-ctlr). TASK-39141's notes list every node with its failure. Checked on the round-12 stack 9091820bf2, 2026-11-08.

**The 86 nodes.**
- TASK-39141 put `test_library_shell.py` and `test_library_ingest_flow.py` under the `bootstrap_profile` marker. 86 nodes then fail on their own assertions: 83 in `test_library_shell.py` and 3 in `test_library_ingest_flow.py`.
- Each one failed in two whole-list reruns and alone. 90 of the 98 head reds, including all 86, also fail on base 8b762b52fa under the same marker.
- TASK-39141's notes group them by failure. Examples: `NoMatches`, width-matrix `29 == 24` and `36 == 31`, note-editor regions, footer copy, CSS rule pins, mocks never called.
- Two of the 86 passed once at load ~100: `test_background_recompose_focus_fallback_stays_inside_screen_content` and `test_library_onboarding_positive_wins_while_another_owner_hangs`.
- 44 more nodes are intermittent under load; they are listed there too.

**A shared-profile question.**
- Every `bootstrap_profile` test in one xdist worker shares one profile: `XDG_DATA_HOME` is `_BOOTSTRAP_DATA_HOME`, and HOME is under `_BOOTSTRAP_CONFIG_ROOT` (`Tests/conftest.py:1860-1874`).
- Only the config file's bytes are restored after each test (`_restore_bootstrap_profile_config`, `:1312`, TASK-36478). Databases one test writes are still there for the next test in that worker.
- A review saw one "stale" node pass. So some stale or intermittent nodes may depend on what earlier tests left behind; counts such as `Conversations (2)`, `'0 of 0'` and `assert 1 == 0` are candidates.

**Overlap and lanes.**
- Some nodes overlap open tasks. For example, `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back` matches TASK-34000.60 (To Do).
- No PR lane runs either file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 86 nodes passes, or is retired with the commit that changed the behaviour it pinned named in the notes; each red is shown first with the same command
- [ ] #2 Each node is classified as test drift, a product defect, or a dependence on what earlier tests left in the shared bootstrap profile, with the evidence named (for example a run in its own process, or with the data home emptied); a product defect gets a test that is red before its fix
- [ ] #3 Where a node overlaps an open task (such as TASK-34000.60), the notes say which task owns it
- [ ] #4 Both files pass 3 of 3 whole-file runs at `-n 4`, or the notes list each node still failing with its own failure
<!-- AC:END -->
