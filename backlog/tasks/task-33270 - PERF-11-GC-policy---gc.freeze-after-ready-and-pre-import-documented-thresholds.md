---
id: TASK-33270
title: 'PERF-11: GC policy - gc.freeze after ready and pre-import, documented thresholds'
status: Done
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
assignee:
- '@claude'
updated_date: 2026-09-29 17:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing in the app calls gc.freeze() or tunes thresholds. Automatic gen-2 collections take 130-871 ms and land on about every 2nd-4th screen switch (heap 0.57M to 1.6M objects after a tour); 4 gen-2 collections (517 ms) run during mount. gc.freeze() after boot measured at about 0 ms. TASK-31966 requires an ADR for a global GC policy (owner decision D3). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-11; every issue with file:line is listed under PERF-11 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An accepted ADR records the GC policy
- [x] #2 The boot heap is frozen after _ui_ready and after the screen pre-import pass
- [x] #3 Over an 8-destination tour, the median automatic gen-2 pause falls at least 3x and boot-window gen-2 time halves versus no freeze (measured); the original under-50 ms maximum moves to TASK-33545 (owner accepted the partial win, 2026-09-29)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`freeze_long_lived_heap(reason)` in `Utils/ui_responsiveness.py` runs `gc.collect(1)` then `gc.freeze()` and logs the frozen count at DEBUG. `app.py` calls it right after `_ui_ready` flips and at the end of `_preimport_screens`, so it runs at most three times per process. Policy and measurements are in ADR-198. `Tests/Performance/test_gc_policy.py` pins the collect-then-freeze order and boots the real app on a private profile, asserting more than 100K objects are frozen at `_ui_ready`. Without the change, only 375 were frozen.

**Measured** on a scratch profile at 235x52: boot, then a 24-visit tour of 8 destinations, gen-2 pauses timed with `gc.callbacks`, 2 runs per arm, PERF-05 applied:

| | No freeze | Freeze |
|---|---|---|
| Boot-window gen-2 total | 2.3-3.4 s | 1.1-1.3 s |
| Tour collections | 8 | 18-20 |
| Median pause | 509-663 ms | 155-170 ms |
| Max pause | 750-993 ms | 277-705 ms |
| Total gen-2 time | 4.5-5.2 s | 3.0-4.3 s |

**The original AC #3 (under 50 ms) cannot be met by the freeze alone.** Remaining pause cost scales with the live heap built after boot, about 0.6M unfrozen objects after the tour. It is dominated by:
- Textual `Strip` render lines: each allocates 7 `FIFOCache`s eagerly, so ~50-60K strips are ~0.4-0.5M GC-tracked objects;
- screens that outlive their visit (filed as TASK-33460).

The freeze also makes full collections about 2x as frequent, because CPython's 25% long-lived rule then compares against a small unfrozen old generation. The retained cost of freezing live objects is +13K-54K objects (1-4%).

**Owner decision, 2026-09-29: accepted as a partial win.** When I asked for D3, I claimed the freeze removes the 130-871 ms pauses; the measurement disproved that, and the owner accepted ADR-198 knowing it. AC #3 was revised to the measured improvement, and the under-50 ms maximum moved to TASK-33545. ADR-198 also records how the policy applies to the planned Python 3.13 and 3.14 migration.

Probe: `gc_tour.py` (freeze/nofreeze arms, `PERF11_DIAG` heap census, `PERF11_CHAIN` holder chains). It lives in the session scratchpad only; the method is described in ADR-198 "Measured result".
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
