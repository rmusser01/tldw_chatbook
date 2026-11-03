---
id: TASK-38645
title: 'Console: Send-readiness tests assert before a deferred control-bar sync lands'
status: Done
assignee:
- '@claude'
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 06:19
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
- [x] #1 Every test in the file passes 10 of 10 runs under -n 4 load, because it waits for the sync to land rather than for a longer fixed pause
- [x] #2 Each test still fails at its assertions under a temporary revert of the TASK-33620.15.22 code it pins: the three agreement tests under the 1facb4b8ef/c353b078f6 readiness-rule revert, and the positive press test (green on pre-fix code by design) under removal of 1facb4b8ef's sendable_attachment_provider wiring
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Show the red at dev: the file 10 times under -n 4 with a second -n 4 UI file looping as load; record which assertions fail.
2. In the test file, replace each bare `console._sync_console_control_bar()` with a helper that repeats the call until it returns True (the screen's sync rendered), awaiting a short pause between attempts, bounded by wall clock with a failure message naming the deferral. Send's state is still read straight after the rendering call; every assertion stays as it is.
3. AC#1: 10 of 10 runs under the same load.
4. AC#2: temporarily revert the TASK-33620.15.22 product lines (composer resync reads the label; screen sync's own OR; Workbench reads draft text only and ignores the vision block), show each mounted test red at its assertion, restore.
5. Preflight; commit; notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only change, rebased onto dev 18482c1cf7 (2026-11-03). Dev's PR #3346 (e75333c318) fixed the Workbench test on its own: its two sync sites now call the file-local `_sync_controls`, which repeats the bare sync until it renders and then runs `pilot.pause()`. That PR also added `test_attachment_readiness_waits_for_checked_config_projection`. This task covers the other three sites. The bare `console._sync_console_control_bar()` calls in the either-sync-site, composer-resync and image-only press tests now `await _sync_console_control_bar_now(console)`, the existing helper in `Tests/UI/test_console_native_chat_flow.py` (already used by ten Console test files). It warms the checked readiness display, renders under both config locks, retries until a sync renders, and fails after 15 s naming the last refusal and Send's state. It returns straight after the rendering call, so these three tests read Send with no await after the screen's sync. Every assertion is unchanged; the plan's own retry helper was dropped for this reuse.

The rebase kept dev's `_sync_controls` at the Workbench sites and this helper at the other three; checked and kept. The either-sync-site test compares the screen's paint with the composer's, and `_sync_controls` ends with a pause in which another site could repaint before the read. That did not happen in practice: under the readiness-rule revert below, with `_sync_controls` swapped in at that site, the test still failed at `screen_last == composer_last` (`document-with-bytes`, `(True, True) == (False, False)`) in 3 of 3 `-n 4` runs. The no-await read makes "screen_last is the screen's paint" hold by construction rather than by timing, so there is no reason to change it.

Cause, corrected by probe: in this harness the sync goes through the checked readiness display (the config value is disk-loaded), not the native-lock probe at config_sync.py:70-73. Holding `_config_file_lock()` from another thread left the bare sync rendering (True). Making the display cold (`ConsoleReadinessConfigProjection.value = None`) made it return False and leave Send stale. Under load the likely trigger is the display's 1 s proof expiring, or its refresh not yet finished, before the test syncs; individual failures were not traced. The product converges on its own: an expired proof requests the 0.2 s replay, and a cold refresh publishes a whole sync. No product bug found.

Evidence at dev 18482c1cf7 (2026-11-03 05:59-06:10Z). Each loaded run had `Tests/UI/test_console_send_price.py -n 4` looping in a second background call (18 runs, all green), and other agents shared the 18 CPUs; the 1-minute load average read 28-44 across the runs.
- Red at base, the base file with bare calls at the three sites: 2 of 10 `-n 4` runs red. The composer-resync test failed twice at `_send_state(send) == (False, False)`, reading `(True, True)`. The either-sync-site test failed once at the label check (`image`, `assert None is not None`). The Workbench, press and projection tests passed all 10.
- AC#1: 10 of 10 `-n 4` runs, 5 passed each (the file now has five tests), under the same load (load average 29-41).
- AC#2: each revert was applied temporarily, the file run with `-n 4`, and the revert undone; `git diff -- tldw_chatbook Tests` was empty afterwards. No test failed at the helper's timeout.
  - Readiness-rule revert, the product lines of 1facb4b8ef (composer resync counts any label; screen counts draft or pending image) and c353b078f6 (Workbench reads draft text only and ignores the vision block): 4 failed, 1 passed. The three agreement tests fail at their assertions. Either-sync-site, `document-with-bytes`: screen `(True, True)` against composer `(False, False)`. Composer-resync, stale label: `(False, False) == (True, True)`. Workbench, `image`: `(True, False) == (False, False)`. The press test passes, as it did on pre-fix code. Dev's projection test also fails, at its own "Deferred control replay did not complete" wait: with the vision block ignored, the Workbench never reads blocked.
  - Removal of 1facb4b8ef's `sendable_attachment_provider` wiring: 4 failed, 1 passed. The press test fails at `_send_state(send) == (False, False)`: Send reads `(True, True)`, disabled, while a press would send the image. The either-sync-site (`image`, both sites `(True, True)`), composer-resync and Workbench (`image`) tests fail at their assertions, each reading `(True, True)` against `(False, False)`. Dev's projection test passes.
- Before the rebase (dev 2dcf038cb8, same load), when the helper covered five sites including the Workbench test's two: red at base in 5 of 10 runs (Workbench test 5 times, composer-resync 2, either-sync-site 1); green in 10 of 10, 4 passed each; the same two reverts red at the same assertions. Control for the wait: with the projection's `value = None` set just before the sync, the bare call failed the composer-resync test (`(True, True) == (False, False)`) and the helper passed it.
- AC#2 re-worded by the coordinator after review (2026-11-03). The original wording, "each test still fails against the Send disagreements that TASK-33620.15.22 fixed", cannot hold for the positive press test. That test passed on the pre-fix code by design, as TASK-33620.15.22's merge evidence records. The new wording names the control that does prove the wait does not blunt it: removing the provider wiring.
- The lesson this task adds to `backlog/docs/lessons-testing-evidence.md` still holds at 18482c1cf7, so its wording is unchanged. Dev's projection test agrees with it: a config lock held from another thread defers the bare sync only once the 1 s display proof has expired (the test holds `_settings_rebuild_lock()` and waits 1.05 s before the sync that returns False).
- `./scripts/preflight.sh` passed at the rebased head.

Files: Tests/UI/test_console_send_ready_attachment.py, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
