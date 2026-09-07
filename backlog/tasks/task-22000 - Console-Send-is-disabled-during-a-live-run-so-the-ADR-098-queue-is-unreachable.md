---
id: TASK-22000
title: >-
  Console Send is disabled during a live run so the ADR-098 queue is unreachable
status: Done
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

ADR-098 / TASK-14808 / TASK-15121 established that an accepted live turn does **not** block
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
tests in `Tests/UI/test_console_native_chat_flow.py` pin the ADR-098 behaviour just as
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

- [x] An owner decision is recorded on whether a live durable turn admits a queued follow-up (ADR-098) or blocks Send
- [x] Send's label, disabled state, and tooltip agree with that decision — no "Queue" label on a button that refuses to queue
- [x] Whichever contract loses has its pinning tests updated in the same change, with the decision cited
- [x] `Tests/UI/test_console_native_chat_flow.py::test_console_composer_stop_is_subdued_when_idle` and `::test_console_duplicate_send_during_stream_does_not_break_stop_control` are green without weakening what they assert about the Stop control
- [x] The chosen behaviour is verified live, not only under pytest

## Owner decision (2026-08-24)

**Restore the ADR-098 queue.** A healthy in-flight durable turn does not block
Send: the button re-labels to "Queue" and admits the draft as a FIFO follow-up.
The block stays for a genuinely *unhealthy* recovery owner, which is what
TASK-19900.3 actually needed it for.

## Implementation Plan

1. Re-derive the mechanism on the current dev rather than trusting the filing.
2. Establish what the block was guarding, and prove FIFO queueing cannot race a
   durable commit — including the mid-commit and unhealthy-owner interleavings.
3. Narrow the blocking predicate to unresolved owners only.
4. Update the round1 pinning assertion, citing the decision, and confirm what it
   genuinely protected is still pinned somewhere.
5. Verify live (real `TldwCli`, isolated sandbox), A/B against pristine dev.
6. Mutation-check every new/changed assertion.

## Implementation Notes

`ConsoleChatStore.dispatch_recovery_blocks_submission` now reads the
**presentation** owner (`dispatch_recovery_for_presentation`) instead of the raw
one, so it fires only when `recovery_needed` is True. Nothing invisible can
refuse a send any more — if the user is not being shown a recovery card, the
gate does not exist. Every state the block was genuinely added for carries
`recovery_needed=True` and is untouched:

* a checkpoint restored from a previous app run (`_hydrate_dispatch_recovery`
  stores an owner only when it needs recovery), which would otherwise hit the
  repository's "active dispatch checkpoint" refusal on the next send;
* a live owner whose terminal settlement failed
  (`mark_dispatch_recovery_needed` /
  `_restore_dispatch_recovery_after_settlement_failure`) — its run state is
  `BLOCKED`, which `is_send_allowed` **permits**, so this gate is the only thing
  between the user and a raw `RuntimeError` from a second durable owner;
* `QUARANTINED` ownership that could not be read at all.

**The filing was half right — narrowing the predicate alone does not fix it.**
`2c7fcd200` made two changes to `ChatScreen._sync_console_composer_action_state`,
not one: it added the recovery predicate *and* changed the active-session line
from an assignment (`send_blocked = not queue_presentation.send_enabled`, the
ADR-098 shape where the queue projection is the sole authority) to an `or` that
folds the raw run state back in. Since `not is_send_allowed` is exactly the
VALIDATING/STREAMING/CHECKING_CITATIONS/RETRYING set that
`derive_prompt_queue_presentation` already reads as `occupies_slot`, the only
state that `or` could change was the one ADR-098 exists for. Both halves are
fixed here; each was mutation-proven to be independently load-bearing.

**Proving the queue cannot race a durable commit** (new
`Tests/Chat/test_console_send_gate_queue_race.py`, real SQLite):

* Admission requires `activity.accepted_live_turn`, which is set only by
  `turn_accepted`/`acknowledge_durable_acceptance` — i.e. strictly *after*
  `commit_durable_turn` returns. A test hooks `create_conversation` (which runs
  inside `commit_durable_turn`'s own `BEGIN IMMEDIATE`; the assertion
  `in_transaction is True` proves the window is real) and offers a follow-up
  there: it is refused `REROUTE_NORMAL_SEND`, leaving no entry behind.
* Admission is a memory-only registry write; a queued entry is *submitted* only
  from `_drain_waiting`, which runs after the previous turn reaches a terminal
  status, by which point settlement has already popped the owner. A test admits
  mid-stream and asserts the provider saw exactly one checkpoint row at each
  entry (`checkpoint_counts == [1, 1]`), FIFO order, and zero rows at the end.
* A follow-up admitted mid-run whose owner then turns unhealthy (settlement
  failure) is refused: the queue pauses, and forcing `resume_and_drain` returns
  the entry to the head with `DISPATCH_REFUSED` rather than consuming it.

`test_healthy_durable_owner_is_not_recovery_before_checkpoint_transition`'s
`blocks_submission is True` assertion is now `False`, with the decision cited in
the test's docstring. What it is *named* for (the pre-transition owner is
runtime truth, never a recovery card) is unchanged and still pinned.

### Live verification (real `TldwCli`, headless Pilot, isolated `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` + scratch `[paths] data_dir`)

Mid-run, sampled while STREAMING:

| | pristine dev `a71e62e4b` | this branch |
|---|---|---|
| `send.label` | `Queue` | `Queue` |
| `send.disabled` (with a draft) | **True** | **False** |
| `send.tooltip` | "Wait for the active Console run to finish before sending." | "Send the active Console session draft." |
| `console-send-blocked` | True | False |
| `blocks_submission` | True | False |
| recovery owner | `DISPATCH_STARTED`, `runtime_active=True`, `recovery_needed=False` | same |

After pressing Queue: admitted (`queue_count=1`), draft cleared, and the drain
produced `['first live prompt', 'queued live follow-up']` in FIFO order with the
queue empty and the run COMPLETED. The real profile's `config.toml` and
ChaChaNotes DB were byte-identical before and after every run.

Note on the dev arm: the probe drove admission through
`handle_console_send_message` (the Enter path, which deliberately bypasses the
disabled button), so dev still queued — the defect was that the **button** was
dead and the **tooltip lied**, which is exactly what the table shows.

### Mutation results

| mutation | tests that caught it |
|---|---|
| predicate reads the raw owner again (pre-fix) | `test_healthy_live_owner_admits_a_queued_follow_up_drained_strictly_after`, `test_healthy_durable_owner_is_not_recovery_before_checkpoint_transition`, both ADR-098 UI tests |
| predicate always returns `False` (over-narrowed) | `test_unhealthy_recovery_owner_still_blocks_submission_and_a_queued_turn`, `test_follow_up_admitted_mid_run_is_refused_when_the_owner_turns_unhealthy`, `test_restored_source_owner_refuses_fresh_submit_before_echo_or_acceptance`, `test_continuation_row_read_failure_remains_blocking_until_exact_reread` |
| composer gate re-folds `send_blocked or …` | both ADR-098 UI tests |
| registry admits before a chain exists | `test_queued_follow_up_cannot_be_admitted_while_the_durable_commit_runs` |

Every mutation was applied and reverted in place (no `git checkout`), and the
full set is green again afterwards.

### Shutdown / error walk

* Quit with a queued follow-up pending: the real confirmation reads
  `Live agent runs: 1 / Sessions with queued prompts: 1 / Unsent queued
  prompts: 1`, and `ConsoleRuntime.dispose()` completes cleanly.
* A queued turn whose predecessor fails: pinned by
  `test_failed_accepted_queued_turn_pauses_remaining_without_requeueing_it`
  (green) and by the new settlement-failure test above.

### Modified or added files

* `tldw_chatbook/Chat/console_chat_store.py` — narrowed predicate.
* `tldw_chatbook/UI/Screens/chat_screen.py` — composer gate defers to the queue
  projection for an active session.
* `Tests/Chat/test_console_dispatch_recovery_fix_round1.py` — updated pin.
* `Tests/Chat/test_console_send_gate_queue_race.py` — new (4 tests).
