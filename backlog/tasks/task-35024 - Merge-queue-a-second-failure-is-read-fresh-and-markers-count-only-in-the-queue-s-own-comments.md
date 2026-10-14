---
id: TASK-35024
title: 'Merge queue: a second failure is read fresh, and markers count only in the
  queue''s own comments'
status: Done
created_date: 2026-10-14 19:22
assignee:
- '@claude'
labels:
- ci
- merge-queue
priority: medium
updated_date: 2026-10-14 19:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the final code review of the tldw_server port of this queue (tldw_server PR #3237, 2026-10-14), and present here too. Once a retry marker exists, the queue evicted whenever the failed run was not live or held at that moment. A decision taken on a snapshot older than a racing queue run's successful re-run, or a kick woken while GitHub still showed the old attempt of an accepted re-run, could therefore evict a healthy PR that had failed once. Separately, markers were read from any comment, so anyone who can comment could hide a retry marker and turn a PR's first failure into an eviction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a retry marker present, a PR is evicted only when a fresh read shows the queue's re-run ran (a higher run attempt than the marker recorded) and the required check failed again
- [x] #2 A run live or held again, or a required check green since the decision, stands down; a re-run not visible yet is a strike, never an immediate eviction
- [x] #3 Retry, strike and de-duplication markers count only in comments the queue itself posted
- [x] #4 The spec's retry rule describes both
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the re-run attempt in the retry comment.
2. Replace the marker branch of _retry with a fresh read (_after_retry).
3. Filter comment reads to the queue's own author.
4. Tests for each, mutation-checked; spec row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ported from tldw_server PR #3237 (TASK-13639 there). The retry comment carries a hidden <!-- merge-queue-attempt:N --> note (the attempt re-run, read before the re-run). _after_retry: fresh GET of the run; live or held -> stand down; run_attempt <= recorded -> rerun-unseen strike (warning + one kick, eviction after 10 min); otherwise a fresh read of the required check (counted checks + stand-ins): running, missing or green -> stand down, else evict failed-twice. Old markers without the note read as attempt 0, so behaviour on them is the fresh-read rule. Comments: COMMENTS_QUERY now asks for author { login __typename }; _queue_comments keeps only login github-actions with __typename Bot (verified on this repo's queue comments, e.g. #3070); has_comment, marker_age and comment_once (both reads) use it. Tests: 4 new (stale passed, pending lag, forged marker, failed again), the fake records comment authors and bumps run_attempt on a re-run; 191 passed through the scratch wrapper. Mutation checks, each failing a test: author filter removed, attempt check removed, fresh-state check removed, live stand-down removed. Not ported (separate findings there, owner approved only these two): restart cap for runs that keep being cancelled, license-while-gates-run (not applicable: chatbook has no license context). Files: scripts/merge_queue.py, Tests/CI/test_merge_queue_actions.py, Docs/superpowers/specs/2026-10-03-merge-queue-design.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
