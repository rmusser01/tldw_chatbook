---
id: TASK-34412
title: >-
  Console: detach closed submit tasks from the maintenance ledger at emergency
  shutdown
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-05 19:14'
updated_date: '2026-10-05 20:30'
labels:
  - console
  - resource-ownership
  - bugfix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3024'
documentation:
  - Docs/QA/task-33620.9/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Permanent emergency shutdown promises to drop volatile ownership for submit tasks whose event loops are already closed. The maintenance admission ledger currently retains those same unreachable tasks after the submit registry releases them, preventing collection and contaminating later verification. Preserve the existing fail-closed and live-loop drain contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Emergency detachment removes each exact closed-loop submit task from both volatile task ledgers without trying to cancel or await a closed loop.
- [x] #2 A live-loop peer remains tracked through normal cancellation and awaited shutdown, including when it shares the closed task preparation.
- [x] #3 Deterministic closed/live controls retain original cleanup and recovery assertions and raw warning/resource outcomes without blanket clears, diagnostic suppression, or a global lifetime policy change.
- [x] #4 The test-owned emergency abandonment retains the actual default pending-task diagnostic exactly once, proves scoped ContextVar restoration, rejects handler/unraisable/never-awaited failures, and leaves normal live-loop controls unchanged; this does not claim production terminal-task cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the complete and isolated failure receipts, then strengthen the actual closed-submit and mixed closed/live regression assertions to expose maintenance-ledger ownership before changing production. 2. In the existing closed-submit detachment boundary, remove only each exact task already proven to have a closed loop from the maintenance ledger alongside its submit/preparation entries. Do not clear live tasks, detach callbacks, change task state, or suppress emergency diagnostics. 3. Run targeted RED/GREEN and existing shutdown/maintenance controls, static and artifact guards, independent review, and record separate warning/resource results before publishing to draft PR3024. ADR required: no. ADR path: N/A; existing ADR085/120/198 apply. Reason: restore an existing exact-owner emergency-detachment contract with no new service, storage, runtime boundary, or global lifetime policy. Follow-up test ownership: retain the warning-as-error original RED and failed custom-handler context experiment. Only the deliberately abandoned fixture captures its public Task Context and runs its existing final collection there; observe the installed default handler through exact first-line log counts to avoid Python custom-handler context re-entry. Retain every COMMITTING, ledger, provider and weakref assertion, verify both scoped bindings restore, reject every captured warning, and treat unraisable exceptions as errors. Run original live/closed and broader affected controls with strict physical census and independent review. No production context/reset, task state, coroutine close, collector setting, foreign-owner or emergency-policy change. This is test-only ownership, not production clean-shutdown qualification; no new ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one exact Task-key removal from maintenance admission in the existing closed-loop submit detachment branch. Strengthened actual closed-submit and mixed closed/live controls produce RED: 2 failed, 1 warning in 0.98s. Initial targeted shutdown/durable/maintenance controls: 25 passed, 2 ContextVar warnings in 9.68s; strict exit 1 retains up to 48 SQLite descriptors. These historical failed receipts remain. Follow-up deliberate-abandonment warning-as-error RED: 1 failed in 1.17s. Same-context/custom-handler probe fails the required real diagnostic in 0.59s; not installed. Actual-default forwarding probe passes0.81s. Test-only public exact Task Context and default-handler observation retain all original COMMITTING, sidecar, ledger, provider and weakref assertions; exact first-line/count equality rejects fallback errors, both bindings restore, captured warnings fail and unraisables are errors. Original live/closed GREEN15pass4.98s; complete preparation/durable non-retention GREEN114pass76.12s, no warnings, strict exit0, zero DB files at all teardowns. Original1000-send case not duplicated here. Changed test format clean, three inherited Ruff findings unchanged. Independent scoped review no actionable issue; QA/incident lesson updated. No production context/reset, coroutine close, Task state, collector settings, foreign-owner or emergency-policy change. Existing ADR085/120/198; no new ADR for this test-only correction. Broader349body-pass receipt remains strict failure with55 acceptance fixture descriptors and two raw pre-fix warnings; its original1000case passed456.20s. Keep In Progress for broader resource, native/Windows/participant/scale and external-review gates; controlled fixture retirement is not production warning-free emergency or terminal abandoned Task evidence.
All eleven artifact guards pass after this correction; whitespace clean. The isolated remaining acceptance module reproduces19 passing bodies12.85s/no warnings but strict exit1 with55 own SQLite descriptors. That separate fixture adoption remains open; no global/foreign teardown is authorized.

## Renumbering provenance

Originally TASK-34402, created 2026-10-05 19:14. Latest-dev integration
introduced the older Vercel vision task, created 2026-10-04 23:43 in commit
d3a625ea8e573cebc87b4a6e78f12405759153e8. Under the TASK-19601 older-arrival
rule, the Vercel task keeps TASK-34402 and this younger Console task moves to
TASK-34412. Its original add commit is
4c67e8242f8ac71b79b901a154b4ddbe620f21ae (replayed as 49995044eded38e73c479e2f608e40c8eb76f852).
Current inbound QA/task/lesson references move with this record; historical PR
comments and raw receipts retain their original identifiers. Scope, acceptance,
implementation and status are unchanged. Refreshed reachable task paths and
active worktrees showed maximum 34411 before this reconciliation; candidate
34412 had no current HEAD/dev or active-worktree Backlog content reference.
<!-- SECTION:NOTES:END -->
