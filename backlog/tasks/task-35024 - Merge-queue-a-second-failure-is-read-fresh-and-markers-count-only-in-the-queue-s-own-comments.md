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
updated_date: 2026-10-17 17:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the final code review of the tldw_server port of this queue (tldw_server PR #3237, 2026-10-14), and present here too. Once a retry marker exists, the queue evicted whenever the failed run was not live or held at that moment. A decision taken on a snapshot older than a racing queue run's successful re-run, or a kick woken while GitHub still showed the old attempt of an accepted re-run, could therefore evict a healthy PR that had failed once. Separately, markers were read from any comment, so anyone who can comment could hide a retry marker and turn a PR's first failure into an eviction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a retry marker present, a PR is evicted only when a fresh read shows the required check failed again after the queue's re-run started, or failed in another run on the head than the one re-run (the head already had its one retry)
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
Ported from tldw_server PR #3237 (TASK-13639 there). The retry comment carries a hidden <!-- merge-queue-attempt:<run id>:<attempt> --> note, read from a fresh GET of the run before the re-run; if that GET shows the run live or held (a racing queue run re-ran it), the queue stands down instead. _after_retry reads fresh, in this order: the run (live or held -> stand down); the required check, counted checks + stand-ins (running, missing or green -> stand down); then a failure in another run than the note's evicts failed-twice at once (the note is per head; a second run comes from close/reopen); no higher attempt than recorded -> rerun-unseen strike (warning + one kick, evict-rerun-unseen after 10 min); a failure that completed before the fresh attempt's run_started_at is the retried one, so the finished re-run has not reported yet -> rerun-unreported strike (its own kind; evict-rerun-unreported after 10 min); otherwise evict failed-twice. A broken run's retry finds its run by the stand-in's link (the run's html_url) before falling back to the newest broken run, which can be one whose check passed and whose queue-tick failed. A marker posted before attempts were recorded (the live queue's) reads as attempt 0: a failure older than the comment is the retried one -> rerun-unseen strike. A note with an attempt alone reads with the run unknown. Comments: COMMENTS_QUERY asks for author { login __typename }; _queue_comments keeps only login github-actions with __typename Bot (verified on this repo's queue comments, e.g. #3070), and every marker read goes through it (marker_age, marker_attempt, both reads in comment_once). A same-repo branch can still post as github-actions through its own workflow's permissions (spec section 9), but that takes write access, which can disarm a PR anyway. Not ported (separate findings there, owner approved only these two): restart cap for runs that keep being cancelled, license-while-gates-run (not applicable: chatbook has no license context). Files: scripts/merge_queue.py, Tests/CI/test_merge_queue_actions.py, Docs/superpowers/specs/2026-10-03-merge-queue-design.md.
Code review round 1 (2026-10-14): 5 minor, all addressed (old markers, note from a fresh GET, direct _after_retry tests; a parametrize id 'live' was silently skipped by the conftest's --run-live gate, known trap, renamed).
Code review round 2 (2026-10-17, after rebasing onto dev 111f8e48fc): 0 critical/important, 9 minor, all addressed. (1) A racing re-run between the listing and the note's GET over-recorded the attempt, and the unseen check ran before the checks read, so a stale decision on a green head could warn and later evict: _retry now stands down on a live/held GET, and _after_retry reads the checks first. (2) The unseen eviction had slug rerun (shared with two other causes): now rerun-unseen, in spec section 7's list. (3) has_comment was dead: removed. (4) The QUEUE_COMMENTER comment overclaimed that only the queue can post as github-actions: reworded, plus a spec section 9 line. (5) The no-result test used an empty check list; realistically the old failure is still listed and that evicted: the run_started_at rule above, test made realistic. (6) Tests added for the note's fresh source, the attempt-0 fallback's eviction branch, and the unseen strike's eviction after the gap. (7) Another run's failure on the same head now evicts failed-twice at once instead of an unseen strike (run id in the note). (8) The strike was keyed per attempt for nothing (one note per head): now rerun-unseen. (9) The author-filter sentence moved from the retry row to section 7's Comments bullet. Tests: Tests/CI/test_merge_queue_actions.py 183 passed through the scratch wrapper; 8 mutants (note from the listing, fallback always true, unseen slug, no live/held stand-down in _retry, other-run check off, run_started_at rule off, unseen before the checks read, author filter off) each fail it.
Re-review of round 2's fix (b0207e784a): 9/9 fixed; 1 important, 3 minor, all addressed. (I1) The run_started_at stand-down had no bound and woke nothing, so a re-run that finished without a newer check (e.g. its aggregate check cancelled) held the front forever with nothing on the PR: now the rerun-unreported strike (warning + one kick, eviction after the gap), which also wakes a tick in the common listing-lag case. (M1) Test for the production eviction path: a broken run failing again with run_started_at == updated_at (pins the strict <). (M2) The broken-run fallback matched the newest broken run, possibly one whose check passed: now the stand-in's link first. (M3) Spec wording: the run id comes from the listing; the other-run rule compares the run the decision names. Tests: 187 passed; 4 more mutants (strike removed, kind shared with rerun-unseen, <= for <, link match removed) each fail it.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
