---
id: TASK-34790
title: 'Merge queue: racing runs leave a failed run per merge and can double-post
  a warning'
status: Done
created_date: 2026-10-11 01:44
labels:
- ci
- merge-queue
priority: low
assignee:
- '@claude'
updated_date: 2026-10-11 04:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each merge into dev wakes two queue runs (the push and the PR close). Seen live on 2026-10-10: the loser's rebase is answered 'Something went wrong while executing your query', which the queue reads as an outage and re-raises, so every merge leaves one failed Merge Queue run (4 that day) although nothing is wrong. Two racing runs also both posted the same rebase-failed warning on #3058, because the comment de-duplication reads and then posts without any guard between.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A queue run that loses the rebase race to another queue run ends without a failed status
- [x] #2 Two queue runs acting on the same head post a given comment at most once
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. A transient rebase error with the head moved: wake a tick and stand down instead of failing the run.\n2. comment_once: after posting, look again and remove this run's copy if a racing run posted the same marker first.\n3. Tests for both; spec and ADR-218.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
1. A transient rebase error whose re-read shows the head moved is a lost race (each merge wakes two queue runs) or a lost response: the run wakes a tick and stands down instead of failing. With the head unmoved it still fails, and still never counts toward an eviction.
2. comment_once looks again after posting; if a racing run's copy of the same marker has a lower comment id, this run removes its own copy. For a moment two copies can exist; one remains. A read-then-post check cannot be made atomic through the API, so removal after the fact is the smallest correct fix.
Tests: test_a_rebase_lost_to_an_outage_wakes_a_tick_only_if_the_head_moved, test_two_racing_runs_leave_one_copy_of_a_comment, and the end-to-end argv test now expects the look-again read. Mutation-checked. Files: scripts/merge_queue.py, Tests/CI/test_merge_queue_actions.py, the spec, ADR-218.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
