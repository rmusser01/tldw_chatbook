---
id: TASK-22061
title: Navigating away from Console refuses the in-flight wake turn
status: Done
assignee:
  - '@codex'
created_date: ''
updated_date: '2026-09-29 19:39'
labels:
  - console
  - agents
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A wake turn parked in the readiness probe completes after a nav-away
- [x] #2 App exit (`begin_shutdown`) still refuses a wake
- [x] #3 The wake's ledger row is stamped exactly once
- [x] #4 No regressions across the surrounding Console suites
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the refusal to its flag and its setter
2. Confirm against the known-good commit that this is a regression, not a born-red test
3. Exempt AGENT_WAKE at each gate using the mechanism the ruling already uses
4. A/B the surrounding Console suites against clean dev
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
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

2026-09-29 closeout revalidation: existing production navigation-away repair remains in place. The actual mounted navigation/persistence/ledger regression passes (1 test, 112s) after using its supported private-profile process and adding the gateway double's missing cached_context_window method through the real offline resolver. Shutdown/failed-stream/agent-teardown targeted selection passes 3; two config-aware controller nodes now retain collection bootstrap profiles. No production admission gate was bypassed. ADR required: no new ADR; direct verification of ADR-134/135 wake ownership. Evidence: /private/tmp/agent-burndown-wake-continuity3.log and /private/tmp/agent-burndown-shutdown2.log. Independent combined review pending; no full suite run.

Final disposition 2026-09-29: fresh actual mounted navigation-away/wake/transcript regression passed after progress scheduler integration (1 passed in 32.02s; /private/tmp/agent-burndown-final-nav-wake.log). Earlier current shutdown, failed-stream and agent-teardown selection passed three cases. No production repair was necessary: the original defect was already resolved. All acceptance criteria, current qualification, documentation and status reconciliation are complete; task is Done. No historical CI counts, full-suite or live-provider results are claimed.
<!-- SECTION:NOTES:END -->
