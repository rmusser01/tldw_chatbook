---
id: TASK-35833
title: The headless-delivery poll test asserts before the Console's attach reconcile ends
status: To Do
created_date: 2026-10-20 12:49
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
- [ ] #1 The test asserts the poll only after the Console's attach reconcile has ended, and its cause is recorded
- [ ] #2 20 of 20 serial single-node runs pass on dev
- [ ] #3 The assertion still fails when the delivery leg arms no poll (a negative control)
<!-- AC:END -->
