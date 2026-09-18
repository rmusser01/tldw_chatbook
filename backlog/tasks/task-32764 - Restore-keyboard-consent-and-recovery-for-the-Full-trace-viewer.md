---
id: TASK-32764
title: Restore keyboard consent and recovery for the Full trace viewer
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 00:54'
updated_date: '2026-09-18 01:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the existing Full trace-view disclosure operable through the real Settings keyboard path, with truthful cancellation, persistence and recovery instead of terminating the app while opening consent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Choosing Full and applying opens the real disclosure without crashing; Escape or Keep Safe leaves the saved and active viewer Safe.
- [x] #2 Explicit View Full confirmation persists the requested viewer choice; capture and PII settings retain their separate semantics.
- [x] #3 Duplicate activation cannot stack confirmations or writes, and failed or stale writes report a truthful result and allow retry.
- [x] #4 The disclosure, both actions and return focus are visible and keyboard-operable at compact and wide widths in both themes.
- [x] #5 Targeted regressions and isolated native journeys record persistence, clean lifecycle and the remaining exchange-capture review scope.
- [x] #6 Full viewer settings apply through a real Console controller without violating its UI-thread ownership; pending writes settle without leaking policy reservations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the real Full-view keyboard consent failure with production styles and the private config writer.
2. Move the existing consent and save flow into one guarded Settings worker; retain the existing policy service and generation checks.
3. Exercise cancellation, explicit confirmation, duplicate activation, failure/retry, stale configuration and consent lifetime through mounted tests; preserve existing structured-outcome coverage.
4. Verify targeted tests, static/artifact checks and native dark/light compact/wide consent/persistence journeys; review the final diff and record the remaining scope.
5. Update task, guide, QA and completion ledger, then save to draft PR 2704 without merging.

ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Restore the already-approved explicit viewer disclosure without changing capture, masking, storage or provider boundaries.

Native review reproduced QueueThreadViolation from offloading the whole controller. Preserve its existing owner-thread policy boundary, offload only config persistence, and add live-controller plus cancellation/reservation regressions before final native verification. This is a direct repair under ADR-097; no new runtime or storage boundary is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the real Full-view keyboard disclosure in a guarded worker, preserved Apply return focus, rejected duplicate/late confirmation, and refreshed capture choices with their policy baseline after reload. Native verification exposed an additional prompt-queue owner-thread violation; the controller now offers an async save entry that offloads only file writing and settles reservations through repeated cancellation, writer failure and opener-session closure while preserving synchronous callers. Corrected obsolete Full capture failure copy with visible retry guidance. No new ADR: direct repair under backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Updated Settings/trace guides and the testing lesson. 59 distinct targeted cases, seven artifact guards, static baseline checks and independent review pass; no full suite. Four final native dark/light compact/wide journeys and twelve inspected captures verify real refusal/retry, stale consent, separate PII choice, Settings recreation and clean lifecycle (11 healthy private databases, unchanged default fingerprints). Evidence: Docs/superpowers/qa/2026-09-17-settings-capture-consent/README.md. Broader provider/Trace and component reviews remain open; PR 2704 stays draft pending explicit user visual approval.
<!-- SECTION:NOTES:END -->
