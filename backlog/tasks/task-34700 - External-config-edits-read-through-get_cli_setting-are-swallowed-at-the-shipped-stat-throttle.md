---
id: TASK-34700
title: >-
  External config edits read through get_cli_setting are swallowed at the
  shipped stat throttle
status: To Do
assignee: []
created_date: '2026-10-10 10:15'
labels:
  - config
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: TASK-26038 promised that an external edit to the config file is picked up on the next read after the 1 s stat throttle. At the shipped throttle it never is when read through get_cli_setting / load_cli_config_and_ensure_existence. The warm path (_warm_config_cache_hit, TASK-32804.1) calls the throttled _external_edit_detected first; when it sees the edit it returns True and restarts the throttle window, so the guarded bootstrap's own check a moment later is inside the window, returns False, and serves the stale cache. _CONFIG_FILE_STAMP is never refreshed, so every later window repeats this and the edit stays invisible until something forces a reload. Tests/test_config_hot_reload.py zeroes the throttle, which hides it (two stats both run).

Evidence (found in the PR #3059 / TASK-33620.15.1 pre-merge review, verified 2026-10-10 on fix/task-33620.15.1-run-start-off-ui): Tests/test_config_hot_reload.py::test_external_edit_is_picked_up_at_the_shipped_throttle, an isolated-profile subprocess (TLDW_CONFIG_PATH bound under tmp_path), edits users_name and reads after each of two 1.1 s windows; observed 'seen=[fixture, fixture] stamp_updated=False'. The same script with the throttle set to 0.0 passes. The test is checked in as an xfail(strict=True) pin naming this task.

Impact: the Console context-policy cadence work stacked on TASK-33620.15.1 refreshes on config generation changes, so it depends on external edits being detected; until this is fixed an externally edited [console] setting never reaches it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At the shipped 1.0 s throttle, the first get_cli_setting read after the throttle window following an external edit returns the edited value
- [ ] #2 After an external edit has been read, the recorded config stamp matches the file, so the same edit is not re-detected (and re-read) in every later window
- [ ] #3 A warm read with no external edit still serves the cache without the locked bootstrap and stats the file at most once per throttle window
- [ ] #4 test_external_edit_is_picked_up_at_the_shipped_throttle passes with its xfail marker removed
<!-- AC:END -->
