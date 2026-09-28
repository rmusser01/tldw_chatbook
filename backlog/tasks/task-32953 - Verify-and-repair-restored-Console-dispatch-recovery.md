---
id: TASK-32953
title: Verify and repair restored Console dispatch recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-25 16:19'
updated_date: '2026-09-25 16:27'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/issues/2708'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate GitHub issue #2708 and verify that Retry anyway and Discard release interrupted chat recovery without losing or duplicating the accepted message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A restored uncertain-delivery owner retains working Retry anyway and Discard controls after an action is refused or rolled back.
- [x] #2 Successful explicit recovery preserves the original user and assistant identities and releases the composer without automatic provider replay.
- [x] #3 Mounted production Console and real SQLite regressions cover refusal then discard, repeated intents, and existing recovery lifecycle behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the stale recovery action latch with a restored SQLite dispatch-started owner and mounted production Console controls.
2. Reconcile local action admission with authoritative recovery refresh without changing recovery vocabulary or durable ownership.
3. Verify successful retry/discard, refusal then second action, duplicate-intent protection, and targeted existing SQLite/UI tests; correct a stale live-queue fixture only if necessary.
4. Record original history, current evidence, limitations, and the missed transition in relevant lessons.
ADR required: no
ADR path: backlog/decisions/079-console-library-conversation-authority.md
Reason: Repair existing explicit source-device recovery behavior and action liveness; no schema, authority, service boundary, or visual design change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the reproduced GitHub #2708 recovery dead end: a restored source-device owner can claim and release Retry or Discard between UI paints, returning the same visible projection while leaving the widget local click latch set. The unchanged presentation previously skipped latch reset, so later buttons looked enabled but never reached the controller. This behavior dates to acd844e8b2 (2026-08-23); the current widget still contained it despite the issue latest-dev comment. The report does not reveal the original first-action refusal cause, so the controlled readiness-refusal and SQLite-rollback reproductions establish the recovery defect rather than attributing every logged trace error to it.

ConsoleDispatchRecoveryRegion now acknowledges a new immutable store snapshot independently of repainting. Polling the identical snapshot still suppresses duplicate clicks before controller admission. No storage, provider, controller, styling, or replay policy changed. New mounted real-SQLite tests restore dispatch_started ownership, refuse Retry or roll back Discard, then successfully discard the same assistant while retaining its user, deleting its checkpoint and releasing the composer without provider entry. A callback test covers unchanged polling, equal-valued completion and in-flight protection. Updated two existing UI fixture assumptions: healthy accepted turns permit queue admission, and the mounted mock gateway supplies cached context capacity. Added the observed coalesced-paint trap to lessons-testing-evidence.md.

Validation: both mounted cases failed before the repair; loading the original HEAD widget through a temporary pytest plugin also makes the new equal-snapshot callback test fail without editing the checkout. Final targeted run: 75 passed in 63.15s across Tests/Chat/test_console_dispatch_recovery.py and four recovery UI modules. Ruff lint and format checks pass for all four touched Python files; scoped git diff --check passes. Self review found no remaining issue. Existing pytest temporary-directory cleanup warnings remain. No full suite, paid provider, real user profile change, staging or commit.

ADR required: no new ADR. Existing backlog/decisions/079-console-library-conversation-authority.md governs the unchanged source-device recovery contract.
<!-- SECTION:NOTES:END -->
