---
id: TASK-38645
title: 'Console: Send-readiness tests assert before a deferred control-bar sync lands'
status: Done
assignee:
- '@claude'
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 00:36
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
- [x] #2 Each test still fails against the Send disagreements that TASK-33620.15.22 fixed (shown under a temporary revert)
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
Test-only change. The five bare `console._sync_console_control_bar()` calls in `Tests/UI/test_console_send_ready_attachment.py` now `await _sync_console_control_bar_now(console)`, the existing helper in `Tests/UI/test_console_native_chat_flow.py` (already used by ten Console test files): it warms the checked readiness display, renders under both config locks, retries until a sync renders, and fails after 15 s naming the last refusal and Send's state. It returns straight after the rendering call, so each test still reads Send with no await in between. Every assertion is unchanged; the plan's own retry helper was dropped for this reuse.

Cause, corrected by probe: in this harness the sync goes through the checked readiness display (the config value is disk-loaded), not the native-lock probe at config_sync.py:70-73. Holding `_config_file_lock()` from another thread left the bare sync rendering (True). Making the display cold (`ConsoleReadinessConfigProjection.value = None`) made it return False and leave Send stale. Under load the likely trigger is the display's 1 s proof expiring, or its refresh not yet finished, before the test syncs; individual failures were not traced. The product converges on its own: an expired proof requests the 0.2 s replay, and a cold refresh publishes a whole sync. No product bug found.

Evidence (dev 2dcf038cb8; each loaded run had `Tests/UI/test_console_send_price.py -n 4` looping in a second background call):
- Red at dev: 5 of 10 `-n 4` runs red. Workbench test 5 times (document-with-bytes `(False, False) == (True, True)`; image `(True, True) == (False, False)`), composer-resync test 2 times (`(True, True) == (False, False)`), either-sync-site test once (`assert None is not None` on the label).
- AC#1: 10 of 10 `-n 4` runs, 4 passed each, under the same load.
- Control for the wait: with the display made cold just before the sync, the bare call fails the composer-resync test (`(True, True) == (False, False)`) and the helper passes it.
- AC#2: temporarily reverted, run with `-n 4`, and restored (`git diff -- tldw_chatbook` empty afterwards).
  - Revert of the product lines of 1facb4b8ef (composer resync counts any label; screen counts draft or pending image) and c353b078f6 (Workbench reads draft text only and ignores the vision block): 3 failed at their assertions, matching TASK-33620.15.22's notes. Document-with-bytes screen `(True, True)` against composer `(False, False)`; stale label `(False, False) == (True, True)`; Workbench image `(True, False) == (False, False)`.
  - The positive press test passes on that revert, as it did on pre-fix code: it pins "a ready Send sends", not a disagreement. It fails under the revert of 1facb4b8ef's `sendable_attachment_provider` wiring at `_send_state(send) == (False, False)`: Send reads disabled while a press would send the image. That revert turns all 4 red.
- `./scripts/preflight.sh` passed.

Files: Tests/UI/test_console_send_ready_attachment.py, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
