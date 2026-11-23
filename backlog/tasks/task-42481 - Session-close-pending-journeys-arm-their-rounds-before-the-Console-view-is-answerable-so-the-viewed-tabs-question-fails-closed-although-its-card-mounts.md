---
id: TASK-42481
title: "Session-close pending journeys arm their rounds before the Console view is answerable, so the viewed tab's question fails closed although its card mounts"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 12:26
updated_date: 2026-11-23 13:08
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_session_tab_close.py
- tldw_chatbook/Chat/console_interrupt_rounds.py
- tldw_chatbook/Chat/console_runtime.py
- backlog/decisions/314-indefinite-human-approval-waits.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in the round's final re-check at the stack top `fu/r22-keychain` 5e53cb455e (dev fe8c87e3e2 plus the round's ports). Paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-42424 fixed one way `Tests/UI/test_console_session_tab_close.py::test_session_close_pending_journeys` fails with `closing the background tab answered the viewed tab`: the viewed tab's question ran its arm-time liveness check under the Close dialog. The test now waits for that question's card and asserts the question is still live before it opens Close. The re-check failed once anyway, with the same message at line 1174 and the sibling round's result `{'answered': False, 'reason': 'cancelled'}`. The card had mounted and the pre-Close `assert not sibling.done()` had passed. So the card wait does not prove the check passed, or a second path settles the viewed tab's round. CI showed this assertion 29 times before TASK-42424's fix existed, so some of those failures may come from this path.

If closing a background tab can cancel another tab's pending decision, a user loses a question or approval they never answered. If it is a harness artefact, the required UI lane keeps failing for nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 What settles the viewed tab's round as `cancelled` during a background tab's Close is recorded with stack evidence, and named as a product defect or a harness artefact
- [x] #2 The cause is fixed where it lives (product code for a defect, test synchronisation for an artefact), with every existing assertion kept, and the failure is shown red before the fix and green after
- [x] #3 The journey passes 20 runs of its file at `-n 4` and 5 serial runs, under load
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read `run_round`'s wait loop, `_is_session_cancelled` and every writer of the events and state they read (`cancel_event`, `visit_event`, `revoked`, the arm-time `app_gone`).
2. Catch it with a scratch probe (not in the repo), loaded as a `sitecustomize` in the private-profile child: log each round's arm, its arm-time view read, its outcome with `run_round`'s locals, every `set()` of its events with a stack, every sweep, every `_reconciled_view` change, every question-card setter call with a stack, and every screen push.
3. Reproduce under natural load and with a forced delay; keep full logs, the child's included.
4. Decide product defect or harness artefact, and fix it where it lives.
5. Soak: the file 20 times at `-n 4` and the journey 5 times serially, under load; `Tests/Chat/test_console_runtime_lifetime.py` at `-n 4`; preflight; ruff F; size ratchets for any product module touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict: a harness artefact, not a cross-session defect.** Nothing in the Close touches the viewed tab's round. Its own arm-time liveness check (TASK-41637, `app_gone` in `InterruptRoundHost.run_round`) fails it closed, because the round arms before the Console view is answerable. `_is_session_cancelled` is not involved. The loop's `cancelled` comes from the `app_gone` term of the same condition (`if app_gone or _is_session_cancelled(...)`).

**Mechanism.**
- `_mounted_console` returns once the composer exists. That is before the Console's attach reconciliation (`ChatScreen._reconcile_console_after_attach` → `ConsoleRuntime.finish_view_reconciliation`) sets `_reconciled_view`. Until then `has_answerable_view()` is False.
- If the viewed tab's question arms in that gap, `run_round` reads `_approval_view_is_detached()` as True. With the harness's never-run `TldwCli`, `app_gone` is True, so `run_round` mounts no card and the round settles `cancelled` at its first 1 s poll.
- The reconciliation then mounts the parked card anyway (`finish_view_reconciliation` → `_remount_session_kinds`), so TASK-42424's card wait and its pre-Close `not sibling.done()` pass. The round ends during the Close sequence, and line 1174 reports that the Close answered the viewed tab.
- TASK-42424's reasoning, "the card mounts only after the liveness check passes", holds only once the view is reconciled. The card has five other writers: the remounts in `new_session`, `switch_session`, `close_session`, `finish_view_reconciliation` and screen resume.
- In the shipped app `controller.app` is running, so `app_gone` is never True and this path cannot occur.

**Stack evidence** (scratch probe, a `sitecustomize` in the private-profile child, never in the repo; full logs and child logs kept under `$FU/tc22-logs/`):
- The arm-time read was True on the worker thread, `view_visible=True`: `test request` → `ConsoleChatController.request_user_questions` → `InterruptRoundHost.request_user_questions:4574` → `run_round:1308` → `_approval_view_is_detached`. At arm: `app_running=False`, `cancel_event=None`, `visit_event` unset.
- The card was mounted by `_reconcile_console_after_attach` → `finish_view_reconciliation:5735` → `_remount_session_kinds` → `remount_for_session`, about 0.6 s after the arm.
- The outcome was `cancelled` with `app_gone=True`, `revoked=False`, the round's event never set, the visit event unset and no cancel event. The Close sweeps named only the closing session and its run.

**Reproduction.**
- Forced delay at the base: the probe held the first attach reconciliation of one iteration.
  - The chat-create pass, held 0.6 s and 0.75 s: the CI message `closing the background tab answered the viewed tab` at :1174, 2 of 2.
  - Held 2.0 s: `the viewed tab's question card never mounted`.
  - The question pass, held 0.6 s: `the closing tab's decision never told the user it waits`. The same race also hits the closing tab's round, and TASK-42424's notice guard catches it.
  - The first scenario's approval pass, held 0.6 s: the same notice message.
  - Red 5 of 5.
- Natural load did not reproduce it locally: 24 probe runs at load 14-20 all passed. In those runs the reconciliation finished at least 0.2 s before any arm. A heavier attempt (6 concurrent runs beside 20 busy loops under `taskpolicy -b`, load 31) only hit the 180 s timeouts and was stopped.
- Timing signature. In passing runs, the reconciliation finished at least 1.03 s before the Close click in the question pass, and at least 3.29 s before it in the chat-create pass. In the round's failing local log, its resume came 0.34 s before the Close click. In CI job 137297925582 (run 45050963273, before TASK-42424) it came 0.39 s before. In job 137216650072 (run 45016281720 attempt 3), the collapsible restore that follows the reconciliation by 0.1 s came 0.46 s before. The forced-delay failures showed 0.32-0.40 s. The other two CI logs read (jobs 137081859988 and 137190507806) have tails that start too late to show it.

**Fix (test synchronisation).** A new helper, `_await_answerable_view(pilot, controller)`, waits for `not controller._approval_view_is_detached()`, the arm-time check's own input. Both pending-close scenarios call it right before arming. Every assertion is unchanged, including TASK-42424's card and notice waits, whose comment now says why the card is evidence only after this wait. Under the same five holds, the fixed test passed 5 of 5, and the probe saw no arm-time detached read.

**Comments.** The `run_round` guard comment in `Chat/console_interrupt_rounds.py` and ADR-314's TASK-41637 clarification now name both windows: before the attach reconciliation, and under a modal. The ADR also says that a card alone does not prove the check passed. The module grows by 1 line, to 6304; its size-ratchet pin is 6307, so the pin is unchanged.

**Siblings.** No other test file pairs the never-run `TldwCli` harness (`app.call_from_thread = host.call_from_thread`) with real rounds. The nine other `Tests/UI` files that arm rounds use a running app, so their `app_gone` is always False. The third `_arm_pending_round` caller arms a chat-create round, which has no liveness check.

**Soak and checks (fixed tree, without the probe).**
- The whole file 20 times at `-n 4` under `taskpolicy -b` beside 6 bounded busy loops (two file runs at a time): 20 of 20 runs, 80 of 80 tests passed, load 13.7-17.7, 120-197 s per run.
- The journey 5 times serially (`-p no:xdist --timeout=180`) under `taskpolicy -b` beside 8 busy loops: 5 of 5 passed, load 11.4-14.2, 51-61 s per run.
- `Tests/Chat/test_console_runtime_lifetime.py` at `-n 4`: 42 passed.
- `Tests/Architecture/test_module_size_ratchet.py` serially: 209 passed.
- `ruff --select F` on the two touched Python files: clean. `Tests/lint/ruff_gate.py`: rc 0, no rule over baseline.
- `preflight.sh`: all checks passed.

No user-visible behaviour changed, so there is no User Guide edit and there are no new census entries. Lesson added to `backlog/docs/lessons-testing-evidence.md`: a card on screen does not show that its round's arm-time check passed. TASK-42424's entry now points to it.

**Files** (fix commit 9421454914). `Tests/UI/test_console_session_tab_close.py`, `tldw_chatbook/Chat/console_interrupt_rounds.py` (a comment only), `backlog/decisions/314-indefinite-human-approval-waits.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
