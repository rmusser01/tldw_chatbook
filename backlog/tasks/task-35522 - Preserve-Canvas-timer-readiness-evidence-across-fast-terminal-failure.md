---
id: TASK-35522
title: Preserve Canvas timer readiness evidence across fast terminal failure
status: Done
created_date: 2026-10-18 15:29
assignee:
- '@codex'
labels:
- canvas
- tests
- uat
priority: medium
documentation:
- Docs/Canvas/UAT_2026-10-18.md
updated_date: 2026-10-18 16:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the real-browser readiness observer so the v1 timer-rate-limit UAT case cannot lose a genuine startup-ready receipt when the runtime reaches its expected terminal quota failure before a later observation. Preserve runtime behavior, exact failure codes and all zero-egress/native-realm assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deterministic real-browser control proves actual approved execution and an observed ready receipt followed by timer-rate-limit failure even when the final observer read is delayed until after failure.
- [x] #2 The shared initial-load helper reports the genuine initial terminal readiness receipt without replacing it with a later mutable status; initial rejection and startup-census failures remain refusals and no generated traffic or parent/native realm mutation is accepted.
- [x] #3 The complete affected Chromium zero-egress module and both v1/v2 adversarial UAT cases pass with unchanged quotas, deadlines, runtime assets and execution policy; optional-browser absences and inherited setup limits remain explicit.
- [x] #4 RED/GREEN evidence and the final UAT report identify the observer defect and maintain the server/client synchronization hold; no product runtime permission or capability changes are introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Reason: test-observer race repair under existing ADR-121 zero-egress/startup proof; no runtime or interface change. 1. Retain passive real MessagePort history and deterministic delayed-final-read evidence from diagnosis. 2. Add a failing control showing the owning load helper returns later failure despite a real earlier ready receipt. 3. Make the smallest shared observer correction using the actual first relevant receipt, preserving startup rejection and exact terminal checks. 4. Run covering real-browser controls and the affected zero-egress module, then independent review and final original UAT rerun. Keep runtime assets and all numerical limits unchanged; no full repository sweep or server work.
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md (existing); no new ADR required.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The test loader returns the retained initial trusted canvas:status receipt instead of a second-round-trip mutable latest status. Added deterministic delayed-observer real-Chromium regressions for both profiles proving execution-started -> ready -> timer-rate-limit, same nonce, startup census, disabled scripts, native-realm integrity and zero generated egress. Audited all29 existing helper owners: one load per fresh page, so no prior-execution history is reused. Genuine2-case RED/GREEN retained; complete zero-egress module35 passed with only absent Firefox/WebKit skipped;15 shared-helper consumers and final original41 UAT passed. Quotas, deadlines, runtime assets and prior refusal expectations unchanged; independent review clean. Existing ADR121 applies: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md. See Docs/Canvas/UAT_2026-10-18.md. Sync remains paused.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
