---
id: TASK-38961
title: "Console endpoint-discovery tests race the delayed control-bar replay and the model popover's compose"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-07 07:23
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
- [x] #1 test_refused_chat_settings_test_blocks_console_until_one_retry, test_create_endpoint_with_live_controller_rebase_settles and test_endpoint_command_create_lands_on_a_pair_in_pick_mode each pass 10 of 10 whole-file -n 4 runs at a recorded load of 30 or more
- [x] #2 The refused-settings test waits for the header badge's text rather than a fixed sleep, and still fails when the Retry path's header repaint is removed (shown under a temporary source edit)
- [x] #3 The two popover tests wait for #console-popover-find to mount before reading it, and still fail when the pick-mode find text is not filled in
- [x] #4 The three tests join a PR lane with the added lane time measured, or the notes say why they stay off
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refused-settings test: replace the fixed 0.3 s settle with a bounded wait for the header badge's expected text, after readiness settles.
2. Popover tests: wait for the pushed switcher's `#console-popover-find` to mount before reading it (one helper, used by both).
3. Negative controls: (A) a temporary edit that drops the header's repaint of the Retry's Ready result; (B) pick mode opening with an empty Find.
4. Run the whole file 10x at -n 4 at load 30 or more.
5. Add the three tests at the UI census tail with the measured serial time.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both races were in the tests' waits; no product change. Base 8b762b52fa, 2026-11-07.

**Refused-settings test (AC#2).** `_console_settled` now takes the header badge's expected text, as a function of the settled readiness. It polls (bounded, about 15 s of 0.05 s pauses) until readiness meets the predicate *and* `#workbench-header-status` reads that text. It replaces the fixed `pilot.pause(0.3)` after readiness moved, which lost to `_sync_console_control_bar`'s coalesced replay under load (TASK-38696). The refused visit waits for `Not ready · refused :9099`; the Retry visit waits for `Ready · reachable HH:MM` from the settled readiness's `observed_at`. Negative control (A), a temporary edit in `WorkbenchHeader`'s sync that skips the header's write of a `Ready · reachable` status: the test failed with "Console readiness or its header badge never settled: 'Not ready · refused :9099'" (load 46-47).

**Popover tests (AC#3).** `_mounted_pick_popover(harness, pilot)` waits until the pushed `ConsoleModelPopover`'s `#console-popover-find` is mounted, not just pushed, and returns both. The two create tests read Find through it. Negative control (B), pick mode opened with `query=""` in `model_switcher`: all three nodes red (`'' == 'New live endpoint '`, `'' == 'Llama local 2 '`, `'' == 'Command box '`, load 46).

**10 whole-file runs at `-n 4` (AC#1).** Load 45-161 (one-minute averages 45, 58, 70, 62, 154, 161, 137, 121, 96, 98). `test_refused_chat_settings_test_blocks_console_until_one_retry`, both cases of `test_create_endpoint_with_live_controller_rebase_settles`, and `test_endpoint_command_create_lands_on_a_pair_in_pick_mode` passed 10 of 10. Other failures, none of them in these tests: `test_shared_evidence_change_refreshes_the_console_once` 10 of 10 (TASK-35567 owns it); `test_create_endpoint_returns_to_responsive_settings_with_settled_evidence[outcome0]` 2 of 10, the same push-before-compose race on `#endpoint-template-name` and `_pick_in_pick_mode` (proposed follow-up).

**Lane (AC#4).** UI census tail, 3 entries: the refused test, the create test (both cases in one entry, since its first case id holds spaces), and the command test. Serially at load 132-178: 30.5 s, 29.1 + 30.8 s, and 24.5 s.

**Files.** `Tests/UI/test_console_endpoint_discovery.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
