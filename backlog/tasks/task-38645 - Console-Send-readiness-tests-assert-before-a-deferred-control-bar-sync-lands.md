---
id: TASK-38645
title: 'Console: Send-readiness tests assert before a deferred control-bar sync lands'
status: To Do
created_date: 2026-11-02 21:52
labels:
- follow-up
- tests
- console
- ci-flake
priority: medium
references:
- Tests/UI/test_console_send_ready_attachment.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-33620.15.22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). `Tests/UI/test_console_send_ready_attachment.py` (TASK-33620.15.22) is a whole-file UI census entry, so a failure fails the required check.

**Cause.** The tests call `console._sync_console_control_bar()` and assert Send's state after one `pilot.pause()`. When `run_console_config_sync` cannot take the config locks without blocking (Console_Modules/config_sync.py:70-73), that call returns False. The refresh then runs on a 0.2 s timer (chat_screen.py:20932-20957, `set_timer(0.2)` at :20872). A diagnostic hook logged `_sync_console_control_bar -> False` just before a failing assertion, which read `(True, False)` against `(True, True)`. In passing runs the same call returned True. This is a test-side race; the product converges 0.2 s later.

**Evidence.**
- **CI:** `test_the_workbench_send_action_agrees_with_the_composer_send` failed three times in UI Fast Lane (23), each on attempt 1, each with `(True, True) == (False, False)`:
  - run 41002435311, 2026-10-30, green on rerun;
  - run 41225403011, 2026-10-31;
  - run 41230425254, 2026-10-31.
- **Local, dev 88af9f9379:**
  - Plain runs: the workbench test failed 1 of 3.
  - Under `-n 4` load: 7 of 12 runs had a red test in this file, and three of its four tests failed at least once:
    - the workbench test, 6 times;
    - `..._composer_resync_reads_what_is_staged...`, 4 times;
    - `..._same_send_readiness_from_either_sync_site`, 3 times.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in the file passes 10 of 10 runs under -n 4 load, because it waits for the sync to land rather than for a longer fixed pause
- [ ] #2 Each test still fails against the Send disagreements that TASK-33620.15.22 fixed (shown under a temporary revert)
<!-- AC:END -->
