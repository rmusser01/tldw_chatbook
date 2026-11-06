---
id: TASK-38958
title: "Console Inspector readiness-word test reads the header one pause after a single hand-called poll, so it fails under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 05:14
labels:
- follow-up
- tests
- flaky
- console
priority: medium
references:
- Tests/UI/test_console_switcher_local_probe.py
- tldw_chatbook/UI/Screens/chat_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**The flake.** `Tests/UI/test_console_switcher_local_probe.py::test_the_inspector_setup_row_and_send_hint_read_the_readiness_word` fails in both of its cases, `[refused]` and `[key-rejected]`. The UI PR lane runs the whole function (`scripts/ui_pr_gate_census.txt:305`).
- **CI.** It failed on PR #3405, UI lane shard 18/32, 2026-11-04 09:59Z, with `AssertionError: assert 'Not ready · refused :9099' == 'Ready · not tested'` at `:1123`.
- **Local, through the scratch-profile wrapper:**
  - alone, 3 of 3 passed at load 12-15;
  - whole file with `-n 4`, 4 of 10 runs failed at load 34-64 (`[key-rejected]` in runs 2 and 10, `[refused]` in runs 3 and 4).
- Every failure is the same assertion at `:1123`. The readiness value and the readiness word are right; only the painted header `#workbench-header-status` is stale.

**Likely cause.** The test stops the poll timer, calls `_poll_console_credential_readiness()` once, waits one `pilot.pause()`, then reads the header. The poll can return without running `sync()` at four points:
- the pre-gate (`tldw_chatbook/UI/Screens/chat_screen.py:10779`);
- a missed non-blocking settings or config lock (`:10817`);
- the cold-live-read deferral (`:10819`);
- `run_console_config_sync(..., defer=lambda: None)` (`:10829`).

With the timer stopped, nothing retries. TASK-35574's notes record that this test passed 7 of 10 even with the poll made a no-op, because a queued attach repaint wrote the header. So as written it cannot tell the poll from that repaint.

**Related.** TASK-35574 (To Do) owns `test_a_probe_settling_mid_poll_still_reaches_the_console` in the same file, which uses the same harness and fails the same way. In run 10 above it failed alongside this test. One PR may fix both.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both the [refused] and [key-rejected] cases pass 20 of 20 whole-file -n 4 runs at a recorded load of at least 40
- [x] #2 The test still fails on its assertion when the poll's refresh is removed (a temporary source edit that returns before sync()), with the queued attach repaint drained or excluded so it cannot pass the test on its own
- [x] #3 The cause is recorded: which poll exit, or which unflushed repaint, left the header reading 'Ready · not tested'
<!-- AC:END -->

## Implementation Plan

1. Reproduce at load >= 40 (whole file, -n 4) with a temporary probe that records, per case, which poll exit ran, what the control bar returned, the readiness display's age, and the header straight after the poll and after the pause.
2. Fix the test from that cause, reading the header straight after the poll so a queued repaint cannot stand in for it.
3. Control: the poll returns before sync() (temporary source edit); the test must fail on its assertion.
4. 20 whole-file -n 4 runs at load >= 40; check whether the same change covers TASK-35574's test.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (AC#3): none of the four filed exits.** A temporary probe in the test recorded several things around the hand-called poll: whether the pre-gate matched before the poll and was committed by it, the cold-read deferral, every `_sync_console_control_bar()` result, each `run_console_config_sync` and `_checked_display_status` result, and the readiness projection's age and key. It also read `#workbench-header-status` straight after the poll and again after the pause. I ran it on 10 whole-file `-n 4` runs at load 40-51, 20 cases in all. Both failures were `[key-rejected]`, in runs 2 and 7.

In each failure:
- The poll got past the pre-gate and the lock and cold-read exits, ran `sync()`, and committed its pre-gate.
- `_sync_console_control_bar()` returned False, because the checked readiness display was 1.071 s old against its 1.0 s life (`max_age`). `_checked_display_status` returned False.
- `run_console_presentation_sync` then requested the coalesced 0.2 s replay (`_console_control_bar_sync_scheduled` True).
- The header read `Ready · not tested` both straight after the poll and after the test's one `pilot.pause()`: the replay had not yet run.

In the 18 passing cases the display was 0.17-0.58 s old, the bar returned True, and the header changed before the poll returned. The pre-gate commits even when the bar defers, so calling the poll a second time would not have retried. The slow part was the test's own setup under load (mount, attach wait, readiness and evidence reads), which outlived the display.

**Fix (test only).** A new helper, `_fresh_readiness_display`, awaits `ConsoleReadinessConfigProjection.for_screen(console).warm()` until it returns True, right before the poll. With a fresh display, the poll paints the header itself before it returns. The test now reads `#workbench-header-status` straight after the poll, before any pause, so a queued repaint (the attach's `_sync_native_console_chat_ui`) cannot stand in for it. The other Inspector reads still follow the pause, as before.

**TASK-35574: the same fix covers it.** `test_a_probe_settling_mid_poll_still_reaches_the_console` uses the same harness and the same single hand-called poll. I applied the same two changes to it: warm before the second poll, and read the header before the pause.
- Under TASK-35574's own control (the evidence version dropped from both the pre-gate and the snapshot, a temporary source edit), it failed 3 of 3, `'Ready · not tested' == 'Not ready · refused :9099'`.
- It passed 10 of 10 serial runs at load 21-43, and every one of the whole-file runs below.

TASK-35574's own file is left for the coordinator.

**AC#2 (control).** `_poll_console_credential_readiness` returned before its lock/`sync()` block (a temporary source edit). Over 3 whole-file runs, both cases and the mid-poll test failed on their assertions every time, e.g. `assert readiness_words(readiness) == word == status` with `'Not ready · key rejected' == 'Ready · not tested'`. The same edit against the original tests failed 14 of 15 case-runs and passed 1: the queued repaint painted the header during the pause.

**AC#1.** I ran 20 whole-file `-n 4` runs, then 3 more for the runs that started below load 40 (33.6, 39.1, 34.3). All 23 passed, 36 passed each, in 38-141 s. Twenty of the 23 started at a 1-min load of 41.5-70.0.

Files: `Tests/UI/test_console_switcher_local_probe.py`. Lesson added to `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
