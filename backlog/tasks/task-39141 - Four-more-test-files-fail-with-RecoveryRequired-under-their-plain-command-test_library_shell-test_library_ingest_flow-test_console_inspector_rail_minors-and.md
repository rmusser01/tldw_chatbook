---
id: TASK-39141
title: "Four more test files fail with RecoveryRequired under their plain command: test_library_shell, test_library_ingest_flow, test_console_inspector_rail_minors and test_persistent_terminal_privacy_boundary"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- test-infra
priority: medium
references:
- Tests/UI/test_library_shell.py
- Tests/integration/test_library_ingest_flow.py
- Tests/UI/test_console_inspector_rail_minors.py
- Tests/Architecture/test_persistent_terminal_privacy_boundary.py
- Tests/UI/app_factory.py
- Tests/conftest.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. These files are outside the list TASK-38111 (round-11 branch fu/r11-recovery) fixes. Runs were at load 33-54.

| File | Run on dev | Result |
|---|---|---|
| `Tests/UI/test_library_shell.py` | 848 nodes collected; a 4-node sample | Sample: 4 of 4 `RecoveryRequired`, through `Tests/UI/app_factory.py:130` (`load_settings()`). TASK-38111's junit has 785 red: 779 `RecoveryRequired` and 6 on their own assertions (`test_ingest_cta_uses_one_canonical_label_everywhere`, `test_library_export_repeat_press_keeps_the_first_origin`, two generated-stylesheet nodes, the rail scroll CSS node, the note CSS node). The file already has 5 `bootstrap_profile` and 1 `private_profile_test` tests. |
| `Tests/integration/test_library_ingest_flow.py` | 13 nodes | 2 failed, 9 errors, 2 passed. All 11 reds are `raw_source_selection_changed`, through the `library_screen` fixture's `_build_test_app()`. |
| `Tests/UI/test_console_inspector_rail_minors.py` | 5 nodes | 4 failed, all `raw_source_selection_changed`. It imports focus-carrier helpers, but TASK-38111's module marker does not carry over. |
| `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` | the file | 1 red: `test_app_owned_terminal_session_leaves_live_console_boundaries_free` fails in `_build_test_app` with `RecoveryRequired`. It passes under a `bootstrap_profile` marker. |

- None of the four files runs on a PR lane.
- About 795 nodes check nothing in a local one-file run, so a regression in these files is invisible locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each file passes, or fails only on its own assertions, under its plain one-file command on a clean checkout; the before and after node counts are recorded, with the nodes still red listed by their own failures
- [ ] #2 Each file's cause is read from its own traceback and named in the notes
- [ ] #3 No test binds config to the real profile, and RecoveryRequired admission behaviour is unchanged
<!-- AC:END -->
