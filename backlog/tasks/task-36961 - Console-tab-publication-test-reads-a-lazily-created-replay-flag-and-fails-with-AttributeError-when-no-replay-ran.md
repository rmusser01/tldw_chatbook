---
id: TASK-36961
title: >-
  Console tab-publication test reads a lazily created replay flag and fails with
  AttributeError when no replay ran
status: Done
assignee:
  - '@claude'
created_date: '2026-10-26 02:10'
updated_date: '2026-10-26 03:24'
labels:
  - console
  - tests
  - flaky
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while re-verifying TASK-33620.15.4 after its rebase onto dev 7a5e2513c5 (2026-10-25, host load average ~33). Tests/UI/test_console_tab_publication_maintenance.py::test_admitted_real_tab_publication_rechecks_replaced_controller waits for 'not console._console_control_bar_replay_whole_sync', but ChatScreen never initialises that attribute: only config_sync.replay_whole_console_sync sets it (product code reads it with getattr(..., False)). When the mounted Console runs no whole-sync replay before the wait, the child fails with "AttributeError: 'ChatScreen' object has no attribute '_console_control_bar_replay_whole_sync'". Serial runs (the test alone or its whole file), same host: dev 7a5e2513c5 failed 2 of 6, dev 0225e77310 2 of 3, dev caa01e43d7 1 of 1, dev 5b5920b8e2 / 1381e70735 / 5ff7f7a16f passed (1-2 each); the TASK-33620.15.4 branch failed 3 of 6. The file is in no PR lane, so nothing gates on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes whether or not the Console ran a whole-sync replay before its wait (shown by a run that forces no replay)
- [x] #2 The test still waits for any replay that is pending, so it does not start its lock before the screen's own syncs settle
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on the TASK-33620.15.4 branch (2026-10-25), test harness only. The quiet-wait predicate in Tests/UI/test_console_tab_publication_maintenance.py (_quiet) now reads the replay flag with getattr(console, '_console_control_bar_replay_whole_sync', False), as the file's two other waits and the product code already do.
- AC#1: a scratch probe (deleted before commit) popped the flag off the mounted ChatScreen right before the wait, forcing the no-replay case. Without the fix it failed with AttributeError (1 of 1); with the fix it passed (1 of 1). The whole file then passed 3 of 3 tests under the plain command.
- AC#2: when a replay is pending the flag is set, so getattr returns True and the wait still holds until it clears; that path is unchanged.
<!-- SECTION:NOTES:END -->
