---
id: TASK-40095
title: Keep the Settings snapshot save responsive under UI CPU load
status: In Progress
created_date: 2026-11-09 10:44
labels:
- ci
- settings
- performance
priority: high
references:
- Tests/UI/test_llamacpp_snapshot_settings.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Chat/console_preparation_reads.py
updated_date: 2026-11-09 11:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI30’s original five-second keyboard Save assertion expired while the real Settings worker held both native config locks and was still writing. A controlled run of the original case reproduced that failure under event-loop CPU pressure after keyboard dispatch returned. The exact hosted CPU callbacks were not observed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The same real Save completes at the original five-second assertion under the same CPU pressure, with native lock and write ownership intact.
- [x] #2 The scheduling scope restores the original process interval after physical retirement, and both original keyboard paint/save variants pass without higher limits or changed assertions.
- [ ] #3 Independent exact-commit review and the published head’s own hosted gates verify the fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Reuse the existing process-local, ref-counted GIL handoff policy around one already finite native worker callback. The save owner, atomic config boundary, cancellation behavior, and all deadlines stay as before.

1. Retain actual UI30 failure and compare the uninstrumented gate’s worker phases with an observed local baseline and a single controlled event-loop CPU competitor after Enter returns.
2. Scope the existing GIL handoff owner to the real snapshot save running on its existing thread. Keep the exact native lock, compare-and-save, and worker interfaces.
3. Verify the same pressure control and the original native keyboard cases at the same five seconds. Audit the switch interval before and after the physically retired save and complete exact review and the successor’s own hosted qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrapped the existing native snapshot save in the existing GIL handoff guard on its worker thread. The expected snapshot, native locks, physical write, and original five-second assertion are preserved. The controlled reproduction timed out before the fix. Under the same pressure the corrected write returned in 2.852 seconds, the UI completion arrived in 2.896 seconds, and the switch interval returned to its original value. Both original keyboard variants also passed without the diagnostic load. Exact committed-head review and its own hosted UI30 remain required for AC3.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
