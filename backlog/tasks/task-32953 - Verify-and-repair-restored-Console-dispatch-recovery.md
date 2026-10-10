---
id: TASK-32953
title: Verify and repair restored Console dispatch recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-25 16:19'
updated_date: '2026-10-10 20:22'
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

Review follow-up (2026-10-10):
1. Retain token-guarded completion and synchronous exception release while acknowledging new equal-valued snapshots; repair callback assertions and add stale/synchronous-failure regressions.
2. Reproduce defects before fixing, run targeted trace/recovery/privacy tests and required preflight, then obtain independent final review.
ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md (32951); backlog/decisions/079-console-library-conversation-authority.md (32953)
Reason: restore existing refusal ownership, safe diagnostics, and callback contracts without changing boundaries or storage.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the reproduced GitHub #2708 recovery dead end: a restored source-device owner can claim and release Retry or Discard between UI paints, returning the same visible projection while leaving the widget local click latch set. The unchanged presentation previously skipped latch reset, so later buttons looked enabled but never reached the controller. This behavior dates to acd844e8b2 (2026-08-23); the current widget still contained it despite the issue latest-dev comment. The report does not reveal the original first-action refusal cause, so the controlled readiness-refusal and SQLite-rollback reproductions establish the recovery defect rather than attributing every logged trace error to it.

ConsoleDispatchRecoveryRegion now acknowledges a new immutable store snapshot independently of repainting. Polling the identical snapshot still suppresses duplicate clicks before controller admission. No storage, provider, controller, styling, or replay policy changed. New mounted real-SQLite tests restore dispatch_started ownership, refuse Retry or roll back Discard, then successfully discard the same assistant while retaining its user, deleting its checkpoint and releasing the composer without provider entry. A callback test covers unchanged polling, equal-valued completion and in-flight protection. Updated two existing UI fixture assumptions: healthy accepted turns permit queue admission, and the mounted mock gateway supplies cached context capacity. Added the observed coalesced-paint trap to lessons-testing-evidence.md.

Validation: both mounted cases failed before the repair; loading the original HEAD widget through a temporary pytest plugin also makes the new equal-snapshot callback test fail without editing the checkout. Final targeted run: 75 passed in 63.15s across Tests/Chat/test_console_dispatch_recovery.py and four recovery UI modules. Ruff lint and format checks pass for all four touched Python files; scoped git diff --check passes. Self review found no remaining issue. Existing pytest temporary-directory cleanup warnings remain. No full suite, paid provider, real user profile change, staging or commit.

ADR required: no new ADR. Existing backlog/decisions/079-console-library-conversation-authority.md governs the unchanged source-device recovery contract.

Review follow-up (2026-10-10): restored exact recovery-boundary rebinding and private thinking-owner marker comparison, safe numeric/list refusal diagnostics, token-guarded recovery completion and synchronous exception release. Rebase preserves dev's failed-call retry fences and bootstrap imports. Added stale-completion/exception controls; callback assertions now inspect the completion argument. Real mounted tests use the existing per-case private-profile owner, and the trace identity module retains its collection-bound private config source. Fixture drafts survive authoritative projection refresh, and callout assertions wait for actual visible paint.

Defects were reproduced before repair. Final affected identity and mounted surface/dispatch cases: 15 passed. The earlier trace/recovery run passed 39 cases and exposed six later repaired cases; these separate receipts are retained without relabelling them as one green run. Undefined-name checks and scoped diff whitespace checks pass; all derived-artifact preflight checks passed on the rebased branch. Final rebased targeted verification and protected CI remain integration gates. Independent review cleared the prior five repairs; the added owner-marker fix receives a final read before merge. No full suite or live external provider was used. ADR required: no new ADR; existing ADR-097/ADR-079 contracts apply.

Final current-dev review verification (2026-10-10, dev cc46cc7300): the combined nine-file trace/model-retry and mounted recovery selection passes all 45 tests in 125.35 seconds. Derived-artifact preflight and undefined-name checks pass. Independent final review also clears the thinking-owner normalization and its changed-content/changed-owner refusal controls. No full suite or live paid provider. Protected merge remains contingent on exact-head required CI.
Final pre-integration verification (2026-10-10, dev 6ef97bdbd059): all 45 affected tests pass in 91.94 seconds, derived-artifact preflight passes, and all six historical review threads are resolved. Independent review is complete. Final integration still requires the protected CI checks after the preceding PR merges.
<!-- SECTION:NOTES:END -->
