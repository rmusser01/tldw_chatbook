---
id: TASK-15661
title: 'Key the parked approval payload by round id (fleet F7)'
status: Done
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parked approval payload lives in a single slot, so two approval rounds that are parked at the same time overwrite each other. This is pre-existing for sibling sub-agents within one turn and is already documented in the file as an accepted limitation, but cross-turn survivors (PR 3a-1) widen the window in which two rounds can be parked together. Key the payload by round id so each parked round keeps its own.

**Exposure update, 2026-08-13 (fleet PR 3a-2, `feat/fleet-autowake`):** auto-wake adds a machine-initiated turn class (`AGENT_WAKE`) that by design runs in sessions the user is not viewing, so approval rounds raised by a woken turn park by default rather than exceptionally. The overwrite MECHANISM is unchanged (the per-session `_parked_approval_payloads` slot), and the wake never fires INTO a session with a card already pending (busy-session deferral gate, pinned) — but the population of parked rounds grows: a woken turn's tool can park a round while an earlier turn's survivor parks another in the same session, with no user action involved at any point. The fixer should treat wake turns as a first-class producer of parked rounds when testing AC #1/#4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two rounds parked at the same time each retain their own payload
- [x] #2 Answering one parked round does not alter or clear the other's payload
- [x] #3 The accepted-limitation comment in the source is removed rather than reworded
- [x] #4 A test parks two rounds concurrently and fails when the slot is shared again
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #1836 (merge 10509d286, 2026-08-20), which re-keyed the retained-payload maps of ALL THREE interrupt bridges (MCP approvals, skill-install, skill-script) from `session_id` to `round_id`, with the mounted card always the session's FIFO head (oldest-armed round). Five shared generic helpers (`_park_round_payload` / `_head_round_payload` / `_unpark_round_payload` / `_remount_head` / `_head_round_payload_locked`) replace the three per-bridge order-dependent `_clear_pending_*_if_round_is_current` TOCTOU guards: re-deriving the head is a pure function of current state, so the guards' two-part checks became unrepresentable rather than merely unnecessary. The decision still runs inside the `call_from_thread` callable (never a worker-thread snapshot). Production −144 lines net.

AC #3: both accepted-limitation texts (the `request_mcp_approvals` `finally` comment and `remount_pending_approval_for_active_session`'s "Known limitation" paragraph) were removed with the code they annotated. AC #4: `Tests/UI/test_console_parked_payload_rekey.py` (8 tests, written RED first) plus rewritten pinned assertions in the task-581/TASK-910 concurrency suites and the headless-approval suite (`test_two_headless_rounds_share_one_payload_slot_and_only_one_mounts` → `test_two_headless_rounds_each_mount_in_turn`).

Post-review riders (Qodo, same PR): payloads carry `deadline_monotonic` and every late mount (promotion / switch-back / attach) receives a remaining-time snapshot instead of the arm-time timeout; `_remount_head(session_id=None)` resolves the session active at callback time, so a legacy no-session round's card cannot strand after a session switch.

Known gap carried into the Console interaction program's sub-project C (recorded in `Docs/superpowers/specs/2026-08-19-console-user-interaction-design.md`): a non-head round on the ACTIVE session is silent (no card, no toast) until promoted, while a background non-head round still toasts.
<!-- SECTION:NOTES:END -->
