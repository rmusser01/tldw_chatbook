---
id: TASK-34369
title: Avoid redundant workspace registry reads for established Console stores
status: Done
created_date: 2026-10-04 18:43
assignee:
- '@codex'
updated_date: 2026-10-04 19:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Windows UAT pauses while Console control and draft refreshes repeatedly resolve workspace storage even though an established session already owns its context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Established Console stores do not resolve registry workspace context during ordinary getter calls
- [x] #2 Empty-store startup alignment and saved-resume scope remain correct
- [x] #3 Targeted tests and live before-and-after Windows timing verify the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record native Windows timings for established-store getter reads. 2. Add a failing storage-access regression. 3. Resolve workspace context only when creating or aligning an empty store. 4. Verify targeted scope tests and live conversation timing. ADR required: no. ADR path: N/A. Reason: remove a redundant read while preserving existing session ownership and startup alignment policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The store getter resolved active workspace registry context on every control/draft refresh, although established sessions already own their context and never use that result. Resolve it only for creation or empty-store alignment; explicit resume retains its existing safe context until hydration. The two established-store regressions failed before the change and pass afterward; empty startup plus existing reconciliation and mounted retry checks pass (14-case scope/mount run, included in the final 216-case run). Review verified predicate matches the branches consuming the result. Live submit acceptance measured 6.1–7.0s on earlier profiled calls and 4.1s on one final unprofiled call; these are observations, not a controlled benchmark. Broader Windows guarded storage latency remains: that final run recorded a 5.5s event-loop stall and 29.1s to provider entry. This task removes the proven redundant read and does not claim to fix all Windows pauses. No ADR required: no change to storage authority, safety gates, settings, styling or session scope. Modified store getter and three focused regressions.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
