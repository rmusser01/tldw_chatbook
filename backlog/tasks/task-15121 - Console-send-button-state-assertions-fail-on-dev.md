---
id: TASK-15121
title: >-
  Console send-button state assertions fail on dev
status: To Do
assignee: []
created_date: '2026-08-11 05:20'
labels:
  - console
  - tests
  - dev-baseline
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two failures in `Tests/UI/test_console_native_chat_flow.py`, both about the send button's state classes:

- `test_console_composer_stop_is_subdued_when_idle` — expects `console-send-blocked`; the button carries `console-action-disabled console-send-inactive console-send-button console-action-subdued` instead.
- `test_console_duplicate_send_during_stream_does_not_break_stop_control` — expects `button.disabled is True`; it is `False`, with classes `console-send-ready … console-action-primary`.

**Proven pre-existing on dev, not introduced by task-14920's repair**: both were re-run against a pristine `git archive origin/dev | tar -x` tree and failed identically there. They were not part of the 20 that task documented — they appeared in the same file after dev moved 46 commits, so a send-button state refactor landed between the two runs.

Triage is the work: the class vocabulary may have been deliberately renamed (`console-send-blocked` → `console-action-disabled`/`console-send-inactive`), in which case the tests are stale pins; or the button genuinely no longer reaches the blocked/disabled state during a stream, which is a real control regression — the second test's name says the stop control must survive a duplicate send. Do not assume the rename reading: check what the button is meant to do mid-stream before rewriting either assertion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Each failure is classified as a stale class-name pin or a real send/stop control regression, with the commit that changed the behaviour named
- [ ] #2 A real regression is fixed in the product; a rename is followed in the test while preserving the original claim (that the control is genuinely unavailable, not merely styled differently)
- [ ] #3 `Tests/UI/test_console_native_chat_flow.py` runs whole with a READ pass count and no unexpected failures
<!-- AC:END -->
