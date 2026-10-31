---
id: TASK-35311
title: Finalize a DONE reply as complete when shutdown or session close lands after
  run end
status: In Progress
assignee:
- Robert
created_date: 2026-10-18 00:00
labels:
- console
- durability
- uat
priority: medium
dependencies: []
updated_date: 2026-10-31 17:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console UAT 2026-10-17 (`qa/console-uat-2026-10-17/`, origin/dev `6c0bde2437`): assistant
replies ended with `assistant_generation_state='stopped'` instead of `'complete'` when the app
instance shut down soon after the reply finished — content fully persisted and rendered, only
the terminal state wrong. A reply whose app lived longer got `'complete'`.

Cause: the bridge worker returns `RUN_DONE` but the controller task is still suspended at
`await run_provider_worker` or has not reached `mark_message_complete`. `controller.shutdown()`
calls `_signal_stop` (permanently sets the cancel event) then cancels the task; the
CancelledError branch and `_finalize_agent_reply`'s `stopped_now` decide purely on
`cancel_event.is_set()`, discarding the DONE outcome. User-Stop and shutdown are currently
indistinguishable at this seam (citation repair already threads a `cancel_reason`).

Governance: ADR-230 / ADR-094 turn-lifetime context; user-Stop semantics are pinned by
`test_shutdown_stops_and_awaits_active_stream_task` and the durable-turn acceptance
parametrization and must not change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:AC:BEGIN -->
- [x] A cancellation landing after the run outcome is RUN_DONE with reason shutdown/session-close finalizes the message `complete` with full content
- [x] User Stop after a DONE outcome still finalizes `stopped` (existing spec preserved)
- [x] Shutdown of a still-streaming run still finalizes `stopped` (existing spec preserved)
- [x] Shutdown retains terminal-write ownership and citation selection stays on the active stream task
- [ ] Adjacent stop/shutdown/finalize tests stay green
<!-- SECTION:AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test: cancel injected at the finalize window with a DONE outcome and shutdown reason →
   assert `complete` (today `stopped`).
2. Thread a stop reason (`user`/`shutdown`/`session_close`) through `_signal_stop`, recorded
   per session; feed citation repair's `cancel_reason`.
3. CancelledError branch + `_finalize_agent_reply.stopped_now` become reason- and
   outcome-aware; shield the finalize call site (bounded drain).
3. Variant tests for user-stop and mid-stream shutdown.
PR3173 repair (2026-10-31): verify completed-result recovery against current provider and citation custody; retain explicit Stop and live-stream shutdown outcomes; add focused lifecycle regressions for any discovered issue, then test affected contracts. ADR required: no. ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md. Reason: repair within existing terminal and cancellation owners. Owner waived size limits and CI waiting, with independent requesting-code-review before publication.

Review repair plan (2026-10-31): reproduce a provider failure after submit cancellation with the real retained native call; keep Stop prompt and retain/observe its wrapper until terminal using the existing maintenance owner. Verify exception observation and retirement without a full sweep. ADR required: no. ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md. Reason: repair lifetime custody within the current owner.
<!-- SECTION:PLAN:END -->
## Implementation Notes

- A retained completed provider outcome survives late shutdown cancellation.
  Recovery finalizes on the original stream task so citation selection keeps its
  active owner, preserves the original answer and issues no repair request after
  shutdown. The former detached finalizer completed the text but left citation
  selection uncommitted; the new regression was RED before this repair.
- Explicit user Stop remains stopped even if shutdown follows. Unfinished streams
  and current dev's already-committed session-close cancellation gate are preserved;
  a closed chat is never resurrected. Reason-only session-close window coverage is
  distinct from deleting a session through that gate.
- Stop reasons retire with their cancel-event owner; provider exceptions retire
  retained worker entries. Both cleanup regressions were RED before the repair.
- The finalization AC states ownership rather than prescribing shielding: the
  existing ADR-094/230 citation contract requires the active stream task. No new
  ADR is needed. This is repair inside the current cancellation/persistence owners.
- Seven new/extended shutdown tests pass, including user-Stop-then-shutdown and
  actual shutdown with armed citation repair. Thirteen targeted adjacent Stop,
  shutdown and citation controls pass. The two `test_citation_repair_stop_while_
  checking_cancels_before_dispatch` variants fail their five-second observation
  wait on BOTH 57e330e735 and the repair; their failed runs also leave teardown
  warnings. No green claim is made for those runs.
- Controller, shutdown regressions, measured size pin, existing test lane and
  Console architecture docs updated. The owner waived size limits and CI waiting;
  independent review is still pending, so status remains In Progress.

- Independent review exposed an unobserved late exception from the provider wrapper
  after submit cancellation. A held real native call reproduced the loop error while
  Stop had already settled. The wrapper now uses the existing maintenance task owner
  for retention and terminal exception observation; no new owner or blocking Stop
  drain was added. All eight focused lifecycle cases pass (review-provider-green.log).
