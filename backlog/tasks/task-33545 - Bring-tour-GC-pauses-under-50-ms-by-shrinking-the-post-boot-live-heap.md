---
id: TASK-33545
title: Bring tour GC pauses under 50 ms by shrinking the post-boot live heap
status: To Do
created_date: 2026-09-29 20:04
dependencies:
- TASK-33460
labels:
- performance
- memory
- perf-audit-2026-09
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carries PERF-11's original target (TASK-33270 AC #3), which a GC freeze alone cannot meet. With ADR-198's freeze and PERF-05's leak fixes, a 24-visit destination tour still showed automatic gen-2 pauses of 155-170 ms median and 277-705 ms max. What remains scales with the unfrozen live heap built after boot, about 0.6M objects at the end of the tour. It is dominated by two things. First, Textual Strip render lines: each eagerly allocates seven FIFOCache objects, so ~50-60K retained strips are ~0.4-0.5M GC-tracked objects. Second, departed screens that stay alive (TASK-33460). Measure which retained strips are live by design (reusable Chat/Library/Home screens) versus avoidable, then reduce the tracked-object count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A heap census after the standard 24-visit tour attributes the unfrozen tracked objects by owner, and the report is recorded in the task
- [ ] #2 The unfrozen tracked-object count after the tour falls by at least half versus the ADR-198 measurement
- [ ] #3 Over the tour, the maximum automatic gen-2 pause is under 50 ms on the tour probe
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
