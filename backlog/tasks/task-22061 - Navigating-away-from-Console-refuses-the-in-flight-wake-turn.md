---
id: TASK-22061
title: Navigating away from Console refuses the in-flight wake turn
status: In Progress
assignee:
  - '@claude'
labels:
  - console
  - agents
  - regression
priority: high
---

## Description

`ConsoleChatController.leave_console` documents an explicit owner ruling: an
in-flight `AGENT_WAKE` turn is NOT cancelled when the user navigates away,
because "cancelling it would re-create the exact 'only completes if you stay'
gap this arc exists to close" (task-15860 P3b). The method implements that for
its own cancel fan-out — it excludes `_agent_wake_turn_sessions` from the tasks
it cancels — but three later gates re-introduced the refusal one layer down by
reading the per-visit `_shutdown_requested` flag that `leave_console` itself
sets.

Result: a wake that fires while Console is mounted, stalls on the provider
readiness probe (an everyday cold llama.cpp probe), and completes after the
user navigates away is refused, stamps no ledger row, and retries.

## Acceptance Criteria

- [x] A wake turn parked in the readiness probe completes after a nav-away
- [x] App exit (`begin_shutdown`) still refuses a wake
- [x] The wake's ledger row is stamped exactly once
- [x] No regressions across the surrounding Console suites

## Implementation Plan

1. Trace the refusal to its flag and its setter
2. Confirm against the known-good commit that this is a regression, not a born-red test
3. Exempt AGENT_WAKE at each gate using the mechanism the ruling already uses
4. A/B the surrounding Console suites against clean dev

## Implementation Notes

Bisected: the file's four tests were green at `10361e2ad` (2026-08-15). Both
`leave_console`'s `_shutdown_requested.set()` and its prompt-queue tombstone
predate that commit, so neither is the cause; the three gates that read the flag
are all newer.

`begin_shutdown` (app exit) always sets `_disposed` before `_shutdown_requested`,
so `_disposed` alone is the complete "app exit" signal — which is exactly what
task-15860 moved `ConsoleFleetWakeCoordinator._attempt`'s gate onto.

Three fixes:
- the outer `submit_draft` fence and the post-resolution acceptance gate now
  exempt `ConsoleSubmissionOrigin.AGENT_WAKE` from the per-visit flag (both
  already had `origin` in scope; the acceptance gate sits four lines below an
  existing `origin is not AGENT_WAKE` special case);
- both pre-dispatch gates route through a new `_teardown_refuses_turn`, which
  consults `_agent_wake_turn_sessions` — the registry `leave_console` already
  uses to spare wake sessions — because neither reply runner takes `origin`;
- `ConsolePromptQueueCoordinator.turn_accepted` no longer raises "accepted
  queued chain is unavailable" for a chain-less turn that carries no
  `entry_id`. It already tolerated that for MANUAL; the strict branch below it
  is the QUEUED path and requires a matching entry id, which a wake never has.

Also repaired a test defect this exposed: `test_console_store_continuity.py`
asserted `_settle(lambda: gateway.payloads)` after the nav-away, but
`_seed_console` has already sent once, so that assertion could never go red —
it reported the failure one step later as a missing ledger stamp. It now
measures growth against a pre-release snapshot. The file's `_StallingWakeGateway`
also lacked the typed `resolved_destination` that `a26cdafd8` made mandatory;
it now derives one through the production classifier (the TASK-21590 pattern)
rather than hand-building it.

Modified: `tldw_chatbook/Chat/console_chat_controller.py`,
`tldw_chatbook/Chat/console_prompt_queue_coordinator.py`,
`Tests/UI/test_console_store_continuity.py`.
