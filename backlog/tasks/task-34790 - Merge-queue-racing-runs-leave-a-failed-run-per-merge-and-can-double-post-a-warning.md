---
id: TASK-34790
title: 'Merge queue: racing runs leave a failed run per merge and can double-post
  a warning'
status: To Do
created_date: 2026-10-11 01:44
labels:
- ci
- merge-queue
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each merge into dev wakes two queue runs (the push and the PR close). Seen live on 2026-10-10: the loser's rebase is answered 'Something went wrong while executing your query', which the queue reads as an outage and re-raises, so every merge leaves one failed Merge Queue run (4 that day) although nothing is wrong. Two racing runs also both posted the same rebase-failed warning on #3058, because the comment de-duplication reads and then posts without any guard between.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A queue run that loses the rebase race to another queue run ends without a failed status
- [ ] #2 Two queue runs acting on the same head post a given comment at most once
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
