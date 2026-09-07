---
id: TASK-2769
title: Two pre-existing Console test failures on dev
status: Done
assignee: []
created_date: '2026-08-07 06:42'
updated_date: '2026-08-07 19:45'
labels:
  - tech-debt
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while verifying wave 3, both reproduced on clean dev at 22c08f958 in disposable worktrees and therefore NOT wave-3 regressions. (1) Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_restores_draft_when_batch_raises fails deterministically in isolation -- a bare-screen fixture reaching query_one via _sync_console_command_popup. (2) Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_retries_console_follow_after_initial_adapter_failure is a load-dependent pilot.click/pause flake: green in a 10-file slice at both commits, red only in a 30-file run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The generation-actions failure passes in isolation
- [x] #2 The live-work-handoffs flake is either fixed or its load dependency is documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both originally-named failures were already fixed on dev by task-2780. What was
actually red were two *other* tests in the same file, plus the recovery test.

**The two fixture failures were my own regression from wave 3, not
pre-existing debt** -- and I had spent a whole subsequent wave treating them as
baseline noise. The `NO_APP` guard I added to `Tests/UI/console_controller_stubs.py`
refuses to infer a missing `app_instance` (an inferred `None` snapshot is a
silent-default hole), and `_bare_console_screen_for_restore` passes `None` from
three of its six call sites. It now declares the absence explicitly.

**The recovery test had three genuine races**, all fixed: `query_one` was
called inside the retry loop and RAISED on a not-yet-mounted node, making the
first iteration fatal rather than a retry; the gate waited on the label alone,
so a press could land while the control was still disabled; and a fixed
`pause(0.1)` after the press assumed the async handler had finished.

**AC #2 is satisfied by documentation, not elimination.** `pilot.click()`
computes an offset then dispatches mouse events, and the recovery re-renders
the rail between those steps: measured, `pilot.click()` passed 6 of 10 isolated
runs where `press()` passed 10 of 10, while `get_widget_at` resolved to the
button every time -- so reachability is now asserted directly and the button
pressed. Under CPU load the recovery itself can still exceed the wait budget
(7 of 12 at load average ~8 vs 10 of 10 idle; a 12s budget did not help, so it
is stuck rather than slow). That residue points at the recovery retry, not the
test, and is recorded in the test's own docstring rather than hidden behind a
longer sleep.
<!-- SECTION:NOTES:END -->

## Addendum (wave-4 task 1, 2026-08-07)

A **third** pre-existing failure in the same file, distinct from the two above:
`Tests/UI/test_console_live_work_handoffs.py`'s `_bare_console_screen_for_restore()`
builds a screen with no app, so the tests using it fail on dev. Same
hand-built-screen-fixture class this programme has shipped once per wave.
Also confirmed pre-existing:
`Tests/UI/test_console_dictation_streaming.py::test_the_transcribing_indication_reverts_on_a_mid_capture_stop`
is a timing flake (failed 1 of 4 isolated runs on a clean baseline worktree,
passes in a full-file run).
