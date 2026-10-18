---
id: TASK-35552
title: Preserve served Canvas preview across concurrent selection reconciliation
status: In Progress
created_date: 2026-10-18 15:56
assignee:
- '@codex'
labels:
- canvas
- bug
- uat
priority: high
references:
- backlog/tasks/task-35521 - Restore-served-Canvas-terminal-rendering-and-reconnect-UAT-evidence.md
documentation:
- Docs/Canvas/UAT_2026-10-18.md
updated_date: 2026-10-18 16:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A completed assistant revision can race the local served Canvas selection reconciliation and permanently disconnect an otherwise live preview. Preserve the existing fail-closed authority contract while allowing the shell to observe the current authorized selection instead of treating a stale navigation completion as a dead session. Server and client synchronization remain paused.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A same-live-session selection epoch advance during awaited navigation produces the existing stale-selection response and never applies or replays the stale result; the shell can refresh the current authorized revision.
- [ ] #2 Revoked, expired, replaced, unavailable or cross-session authority remains refused; unrelated invalid navigation results are not accepted as successful selections.
- [ ] #3 A deterministic regression fails on the original handler and passes after the bounded correction; connected gateway navigation and live create/update/reopen journeys pass without timing or security relaxation.
- [ ] #4 The deterministic navigation-race regression runs exactly once in a required PR lane and the existing CI contract pins its selection.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md. Reason: routine race-classification correction under the existing ownership and stale-selection contract, not a new protocol, capability, storage or sync policy. 1. Retain natural-input503 and successful-race receipts and trace navigation callers. 2. Add deterministic in-flight selection-advance plus revoked, expired, replaced, unavailable and cross-session negative controls before source edits. 3. Apply the existing independently current selection409 pattern to navigation completion without accepting or replaying stale commands. 4. Run targeted gateway tests, original41-case local UAT and independent review; document evidence. No sync work, installs, full repository sweep or publication.
Gate the quick exact navigation regression node in the existing PR Fast Lane sandboxed invocation, preserving its original source-admission profile and without adding a job or increasing timeouts; update the existing exact argv and Console regression-gate contracts, then run Tests/CI.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
