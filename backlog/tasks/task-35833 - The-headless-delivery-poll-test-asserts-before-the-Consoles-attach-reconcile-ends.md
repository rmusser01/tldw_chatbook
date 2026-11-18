---
id: TASK-35833
title: The headless-delivery poll test asserts before the Console's attach reconcile ends
status: Done
assignee:
- '@claude'
created_date: 2026-10-20 12:49
updated_date: 2026-10-22 20:14
labels:
- follow-up
- tests
- flaky
- console
- ci
priority: medium
references:
- Tests/UI/test_console_runtime_ownership.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll` runs in the required check's PR Fast Lane (admission-sensitive), among dev's own targets. It fails intermittently on dev with "Console opened during a wake delivery with no transcript poll armed: reconciled=False".

- **CI.** It failed on PR #3176 at df4e73c908 (dev 0fef489ece, without #3169; run 39375455061, 2026-10-20 06:01Z), and on PR #3169 at 03dd91f9ef (run 39420530080 attempt 1, 12:31Z).
- **Locally.** Scratch runner, node mode, 2026-10-20 about 12:44-12:47Z: 4 of 10 runs failed at dev 8e879de639 and 2 of 10 at #3169's head, every failure with reconciled=False.

**The cause, inferred from the message and from TASK-35574's measured mechanism.** Only the attach reconcile's view start creates the Console's transcript poll timer. After navigating back to Console, the test waits for `#console-native-composer` and one `pilot.pause()`. That proves the Console mounted, not that its attach reconcile ran, and then the test asserts the timer exists. #3169's b94aece7d6 fixed the same shape in TASK-33005.7's census test by waiting for `_console_attach_reconciled and not _console_attach_reconcile_running` first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test asserts the poll only after the Console's attach reconcile has ended, and its cause is recorded
- [x] #2 20 of 20 serial single-node runs pass on dev
- [x] #3 The assertion still fails when the delivery leg arms no poll (a negative control)
- [x] #4 The same file's other test that reopens a cached Console during a wake delivery (test_public_startup_console_keeps_its_claim_across_navigation) waits for the reconcile to end and the deferred arm to run before both of its poll checks
<!-- AC:END -->

## Implementation Plan

1. Check whether the failing heads (8e879de639, df4e73c908, 03dd91f9ef) contain TASK-35710's attachment wait (885569315d), and run the node 10 times serially at dev 28beeb9ed5 for a baseline.
2. Confirm the cause: with the test temporarily back in its pre-TASK-35710 form (composer selector plus one pause) and a temporary print of the attach flags and the timer at the assertion, show that failing runs have reconciled=False there. Revert.
3. Make the existing `wait_for_attachment` helper wait for the end of the reconcile in b94aece7d6's style (`_console_attach_reconciled and not _console_attach_reconcile_running`), with a message naming that condition. No sleeps.
4. 20 serial single-node runs through the wrapper; the whole file once with `-n 4`.
5. Negative control: temporarily stop the attach view start from arming the poll during a delivery, show the test fails with the poll message and reconciled=True, then revert.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**This task was filed from runs that predate its fix.** All four failing heads (8e879de639, df4e73c908, 03dd91f9ef, and #3176's base 0fef489ece) predate 885569315d (TASK-35710, merged with #3176 on 2026-10-20 at 19:08Z). That commit already made the test wait for `_console_attach_reconciled` at all three visits. At dev 28beeb9ed5 the node passed 10 of 10 serial runs, so dev had 0 failures.

**Cause, confirmed (AC#1).** With TASK-35710's test diff temporarily reverted (composer selector plus one `pilot.pause()`) and a temporary write of the attach flags at the assertion, 3 of 10 serial runs failed with the original message. Every failing run had `reconciled=False running=False sync_complete=False` at the assertion, and every passing run had `reconciled=True`. The instrumentation also showed `same_screen=True` and `view_started=True` on the delivery leg: the reopened Console is the cached screen from the control leg. Its reconcile therefore skips `_start_console_view_after_reconciliation`, and the poll is armed only through the wake hook. That path is `finish_view_reconciliation` → `_rearm_delivery_ui_hook` → `_on_console_wake_delivery_started` → `screen.call_later(_start_transcript_sync_timer)`, so the arm is deferred. A temporary write at the line that sets `_console_attach_reconciled = True` showed the timer still unset there (`timer=False poll_needed=True`) in 5 of 5 runs. Dev's wait could still check in that gap: the reconcile runs inside the screen's after-refresh callback batch, which awaits each callback, so the test task can wake in between. The product is fine: the arm runs one screen message after the reconcile ends. All instrumentation was reverted.

**Fix (test only).** `wait_for_attachment` now waits for `_console_attach_reconciled and not _console_attach_reconcile_running`, in b94aece7d6's style: the same bounded 50 x 0.1 s loop, with a message that names that condition and prints both flags. It then calls the existing `Tests/UI/pump_probe.py::pump_runs(screen, 5.0)`. That helper posts a `call_later` and waits for it with a bound. Because the screen's queue is FIFO, the barrier runs only after the hook's deferred arm. No bare sleeps.

**Evidence.**
- AC#2: the fixed node passed 20 of 20 serial single-node runs through the wrapper, each a separate pytest invocation without xdist (11.6-20.2 s each). The comparison is 0 of 10 failures at dev 28beeb9ed5 and 3 of 10 for the pre-TASK-35710 form.
- AC#3: with `_on_console_wake_delivery_started` returning early (temporary edit, reverted), the test failed with `Console opened during a wake delivery with no transcript poll armed: reconciled=True`. A second control, with `_reconcile_console_after_attach` returning early, failed at the new wait: `Console attach reconcile never ended before the poll check (_console_attach_reconciled and not _console_attach_reconcile_running): reconciled=False, running=False, ...`.
- Whole file with `-n 4`: 86 passed, 1 xfailed (the TASK-32873 strict xfail, already on dev). `Tests/UI/test_console_attach_completion.py`, which imports from this file, and `Tests/Packaging/test_voice_source_digest.py`, which lists it, passed together: 84 passed. Preflight passed. Ruff reports the same findings as dev.

**Review round 1 (AC#4, added from the review of 811d9b3939).** `test_public_startup_console_keeps_its_claim_across_navigation` reopens the retained startup Console, the same `chat` instance, both with nothing delivering and during a stand-in wake delivery. Both returns had the same gap: they waited only for `_console_attach_reconciled`, then checked the timer (`is None`, then `is not None`). Both return waits now go through a local `wait_for_reconcile_end()`. It waits for `reconciled and not running`, with a message naming that condition, then runs `pump_runs(chat, 5.0)`. The initial startup wait has no poll check after it and is unchanged. The reattach in `test_native_acceptance_consumes_only_open_target_revision` builds a fresh ChatScreen, so the reviewer left it alone. Evidence:
- 5 of 5 serial single-node runs pass.
- With `_on_console_wake_delivery_started` returning early (temporary, reverted), the test fails at its poll assertion: `assert None is not None` on `_console_transcript_sync_timer`, line 2215.
- The whole file plus `Tests/UI/test_console_attach_completion.py` with `-n 4`: 97 passed, 1 xfailed (the strict xfail already on dev).
- Preflight passed, and ruff reports the same findings as dev.

The lesson's "separate lesson below" pointer now names TASK-34406.5 ("Attach completion does not retire queued mount refreshes"), since the follow-up paragraph now sits between them.

**Files:** `Tests/UI/test_console_runtime_ownership.py`, `backlog/docs/lessons-testing-evidence.md` (a follow-up under TASK-35710's lesson), and this task.
<!-- SECTION:NOTES:END -->
