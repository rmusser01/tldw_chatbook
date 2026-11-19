---
id: TASK-38961
title: "Console endpoint-discovery tests race the delayed control-bar replay and the model popover's compose"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- tests
- flaky
- console
priority: low
references:
- Tests/UI/test_console_endpoint_discovery.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05 (load 14-65). None of the tests below runs on a PR lane: the census and `derived-artifacts.yml:662-670` list other tests from this file.

**The refused-settings test.**
- `Tests/UI/test_console_endpoint_discovery.py::test_refused_chat_settings_test_blocks_console_until_one_retry` (`:847`) passed 3 of 3 alone and failed 1 of 10 whole-file `-n 4` runs (load 39.6).
- It had already failed 1 of 3 on dev and 1 of 3 on a branch earlier in this burn-down.
- The failure is at `:949`: `#workbench-header-status` still reads `Not ready · refused :9099` (expected `Ready · reachable ...`), although readiness and both rail rows read Ready.
- **Likely cause:**
  - `_console_settled` (`:829-836`) sleeps a fixed 0.3 s after readiness flips.
  - The rail rows repaint inside `_poll_console_credential_readiness`.
  - The header repaints through `_sync_console_control_bar` (`chat_screen.py:20918-20945`), which goes through `run_console_presentation_sync` (`config_sync.py:223-249`). That call can defer and replay after 0.2 s or a backoff (`request_control_bar_sync`, `config_sync.py:199-221`; TASK-38696), which under load goes past 0.3 s.

**A second race in the same 10 runs.**
- In 2 of them (load about 48-52), `test_create_endpoint_with_live_controller_rebase_settles` (`:411`) and `test_endpoint_command_create_lands_on_a_pair_in_pick_mode` (`:528`) raised `NoMatches: No nodes match '#console-popover-find' on ConsoleModelPopover()`.
- Their wait loops (for example `:521-526`) stop as soon as the popover screen is pushed, then query a child that has not mounted yet.

**Out of scope.** `test_shared_evidence_change_refreshes_the_console_once` failed in every run; TASK-35567 (To Do) owns it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_refused_chat_settings_test_blocks_console_until_one_retry, test_create_endpoint_with_live_controller_rebase_settles and test_endpoint_command_create_lands_on_a_pair_in_pick_mode each pass 10 of 10 whole-file -n 4 runs at a recorded load of 30 or more
- [ ] #2 The refused-settings test waits for the header badge's text rather than a fixed sleep, and still fails when the Retry path's header repaint is removed (shown under a temporary source edit)
- [ ] #3 The two popover tests wait for #console-popover-find to mount before reading it, and still fail when the pick-mode find text is not filled in
- [ ] #4 The three tests join a PR lane with the added lane time measured, or the notes say why they stay off
<!-- AC:END -->
