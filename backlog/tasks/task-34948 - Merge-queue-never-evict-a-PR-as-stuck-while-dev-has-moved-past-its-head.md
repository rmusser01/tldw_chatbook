---
id: TASK-34948
title: 'Merge queue: never evict a PR as stuck while dev has moved past its head'
status: Done
created_date: 2026-10-14 13:13
assignee:
- '@claude'
labels:
- ci
- merge-queue
priority: medium
updated_date: 2026-10-14 13:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-10-11 the queue removed #3070 from the line twice as 'green for over 15 minutes but auto-merge did not fire', each time within 35 seconds of the PR ahead of it merging. The queue run woken by that merge read #3070's merge state before GitHub had updated it, so the PR looked up to date although dev had just moved past it. Its last green run was hours old (it had waited in line), so the 15-minute stuck rule fired at once. In the same seconds the merged PR's own queue tick rebased #3070 and approved its CI, which went green with auto-merge already off; the PR then sat unarmed for three days. A PR that is behind dev cannot be stuck: it needs a rebase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A front PR whose required check went green over 15 minutes ago is rebased, not evicted, when dev has commits its head lacks, whatever its merge state reads
- [x] #2 A front PR in that state that is up to date with dev is still evicted as stuck, as before
- [x] #3 A failed check of whether the head is behind dev fails the queue run and leaves the PR armed
- [x] #4 The merge queue spec's decision table describes the check
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests in Tests/CI/test_merge_queue_actions.py (FakeGh answers dev...sha compares).
2. In _decide, when decide_front returns the stuck eviction, ask compare/dev...head; behind_by > 0 -> rebase.
3. Update the spec decision-table row.
4. Run the merge-queue tests and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
One GitHub call (compare dev...head, per_page=1), made only when the queue is about to evict as stuck, so normal decisions cost nothing extra. decide_front stays pure; the check lives in _decide, the impure seam that already reads checks and runs, so dry mode reports the same decision. A compare error propagates (GhError): the run fails and the next event decides again, following the spec rule that transient errors never disarm. Evidence for the cause: #3070 timeline 2026-10-11 16:10:51 force-push by the tick in derived-artifacts run 38152170228 (#3029's head), 16:10:53 evict-stuck comment on the old head from merge-queue run 38154337453 (pull_request closed for #3029, merged 16:10:19), 16:11:00 rebased comment; the 03:54Z eviction followed #3071's merge at 03:53:43 the same way. Earlier stuck evictions (#2874, #3026 on 10-06) were the dispatched-check problem fixed by #3039. Files: scripts/merge_queue.py, Tests/CI/test_merge_queue_actions.py, Docs/superpowers/specs/2026-10-03-merge-queue-design.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
