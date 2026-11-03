---
id: TASK-38651
title: First-mount two-launch tests can boot the next app before load_settings carries the saved rail state
status: To Do
created_date: 2026-11-03 05:51
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_first_chat_first_mount.py
- Tests/UI/test_console_collapsed_inspector_send_layout.py
dependencies:
- TASK-38644
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-38644 implementer (2026-11-03); no failure has been seen yet.

`Tests/UI/test_console_first_chat_first_mount.py` has tests that boot a second app on the same private profile after the first one saves rail state, for example `test_a_per_workspace_edit_leaves_the_global_rail_layout_at_its_defaults` (:252 on dev 18482c1cf7) and `test_untouched_workspace_first_use_survives_later_global_edit` (:456). Between launches they poll the config file on disk. The next app's config, though, is built from `load_settings()`, which is republished just after the file is written. So a short window remains in which the second app can boot with the settings from before the save.

TASK-38644 fixed this same mechanism in the late-opened Inspector parity test by waiting for `load_settings()` itself to carry the stored state (PR #3358).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in the file that boots another app after a save waits until `load_settings()` carries the saved state, not only the file on disk
- [ ] #2 The window is shown: under a temporary source edit that delays the republish, the old wait lets the next app boot stale (red) and the new wait passes
- [ ] #3 The file passes 10 runs with `-n 4`
<!-- AC:END -->
