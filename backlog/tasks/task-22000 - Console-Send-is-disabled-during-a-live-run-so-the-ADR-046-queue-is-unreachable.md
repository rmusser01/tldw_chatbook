---
id: TASK-22000
title: >-
  Console Send is disabled during a live run so the ADR-046 queue is unreachable
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - console
  - regression
  - owner-decision
priority: high
---
## Description

Two shipped contracts now disagree about what Send does while a Console turn is running,
and the losing one is the user-visible half.

ADR-046 / TASK-14808 / TASK-15121 established that an accepted live turn does **not** block
Send: the button re-labels to "Queue" and admits the draft as a FIFO follow-up turn. The
whole queue subsystem is still wired and still renders ("Queue 0/10 · Draining", a queue
region, a 10-entry cap, queue-full copy).

TASK-19900.3's durable-turn work then added `ConsoleChatStore.dispatch_recovery_blocks_
submission`, which `ChatScreen` folds into `send_blocked`. It returns True for a
`DISPATCH_STARTED` recovery owner — including the app's own healthy, in-flight run, whose
state is explicitly published as `runtime_active=True, recovery_needed=False`. The result is
that Send is disabled for the whole duration of every non-ephemeral turn.

The user sees a button labelled **"Queue"** that is greyed out and whose tooltip says
**"Wait for the active Console run to finish before sending."** The label promises the
feature; the state denies it. Whichever behaviour is intended, the two cannot both be.

This needs an owner decision, not a unilateral fix: `Tests/Chat/test_console_dispatch_
recovery_fix_round1.py::test_healthy_durable_owner_is_not_recovery_before_checkpoint_
transition` deliberately asserts `dispatch_recovery_blocks_submission(...) is True` for a
healthy live owner, so the durable-turn programme pinned the new behaviour on purpose. Two
tests in `Tests/UI/test_console_native_chat_flow.py` pin the ADR-046 behaviour just as
deliberately. One of the two sets is now wrong.

## Evidence (live, real `TldwCli`, isolated sandbox, 2026-08-24, dev `d589c56c5`)

A headless Pilot run of the real app with only `chat_api_call` stubbed (slow, 6 s), sampled
mid-run:

```
run_state: ConsoleRunStatus.STREAMING
send.disabled = True
send.label = 'Queue'
send has console-send-blocked = True
send.tooltip = 'Wait for the active Console run to finish before sending.'
dispatch_recovery_blocks_submission = True
dispatch_recovery kind = ConsoleDispatchRecoveryKind.DISPATCH_STARTED
[with draft] send.disabled = True     # typing a follow-up does not re-enable it
```

Sequential sends are fine: after the run completes the recovery owner clears
(`dispatch_recovery_for_session -> None`, `blocks_submission -> False`) and a second message
sends normally. Only the *concurrent* queue affordance is dead.

Introduced by `2c7fcd200` "fix(console): enforce dispatch recovery ownership" (2026-08-23),
the commit that added `dispatch_recovery_blocks_submission` and wired it into
`ChatScreen`'s `send_blocked`.

Surfaced by TASK-21590, which repaired the stale Console send harness; these two tests were
the last of that file's 26 failures and are the only two that are **not** harness staleness.

## Acceptance Criteria

- [ ] An owner decision is recorded on whether a live durable turn admits a queued follow-up (ADR-046) or blocks Send
- [ ] Send's label, disabled state, and tooltip agree with that decision — no "Queue" label on a button that refuses to queue
- [ ] Whichever contract loses has its pinning tests updated in the same change, with the decision cited
- [ ] `Tests/UI/test_console_native_chat_flow.py::test_console_composer_stop_is_subdued_when_idle` and `::test_console_duplicate_send_during_stream_does_not_break_stop_control` are green without weakening what they assert about the Stop control
- [ ] The chosen behaviour is verified live, not only under pytest
