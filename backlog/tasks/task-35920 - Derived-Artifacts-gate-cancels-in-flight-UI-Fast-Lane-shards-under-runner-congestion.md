---
id: TASK-35920
title: Derived Artifacts gate cancels in-flight UI Fast Lane shards under runner congestion
status: To Do
created_date: 2026-10-21 23:27
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed twice on PR #3188 (2026-10-21, runs 39596258765 and its predecessor): under runner congestion, three UI Fast Lane shards (2/9/10) reached 96%+ with zero test failures, then were CANCELLED at the same moment the Derived Artifacts gate job concluded 'The required UI fast lane did not succeed' (12:33:17Z pattern). The eviction comment trail on PR #3188 shows 'required check failed twice' for what is pure wait-timeout behavior — all 15 other jobs green, all DA-internal checks green (census 523/523 at the time). The gate's wait-for-lanes logic appears to cancel stragglers when its deadline hits instead of waiting longer or failing soft. Cost: an armed PR gets evicted for congestion, requiring manual re-arm; on a busy day this can loop. Candidate fix: make the gate's lane-wait timeout configurable/longer under queue depth, or treat cancelled-at-high-completion shards as retryable rather than failed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Wait-timeout mechanism identified in the gate job,Cancellation-under-congestion either waits longer, retries, or fails soft with evidence
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Evidence from PR #3190's eviction (2026-10-22, TASK-35660).** In both runs below, the cancelled shards ran out their own 20-minute job cap before the gate job started, so the gate cannot have cancelled them.
- PR #3190, run 39610893391 attempts 2 and 3: UI Fast Lane (2), jobs 119259185494 and 119280385871, each carry the annotation 'The job has exceeded the maximum execution time of 20m0s', after 20m19s each, at 97% with no failing test. The gate job started afterwards: 14:12:22Z after 14:12:00Z, and 14:53:36Z after 14:48:29Z.
- This task's run 39596258765: jobs 9, 10 and 2 ran 20m01s, 20m16s and 20m19s and ended by 12:31:29Z; the gate job started at 12:32:47Z. (The API lists no annotations for these three jobs now, so the cap is read from their durations.)
- Both runs used the ten-shard deal. TASK-35834's thirteen-shard deal (PR #3182) merged at 15:07Z the same day. Its headroom has since eroded again on dev: TASK-36140.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
