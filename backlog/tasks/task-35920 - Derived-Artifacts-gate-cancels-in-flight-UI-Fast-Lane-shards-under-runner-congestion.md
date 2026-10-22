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

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
