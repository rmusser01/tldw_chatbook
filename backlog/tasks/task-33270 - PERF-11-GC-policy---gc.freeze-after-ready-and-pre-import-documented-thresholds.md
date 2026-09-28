---
id: TASK-33270
title: 'PERF-11: GC policy - gc.freeze after ready and pre-import, documented thresholds'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33264
labels:
- performance
- memory
- adr
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing in the app calls gc.freeze() or tunes thresholds. Automatic gen-2 collections take 130-871 ms and land on about every 2nd-4th screen switch (heap 0.57M to 1.6M objects after a tour); 4 gen-2 collections (517 ms) run during mount. gc.freeze() after boot measured at about 0 ms. TASK-31966 requires an ADR for a global GC policy (owner decision D3). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-11; every issue with file:line is listed under PERF-11 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An accepted ADR records the GC policy
- [ ] #2 The boot heap is frozen after _ui_ready and after the screen pre-import pass
- [ ] #3 Over an 8-destination tour, the maximum automatic GC pause is under 50 ms on the tour probe
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
