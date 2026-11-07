---
id: TASK-34920
title: A shared Dreams source can stay off for good after a partly recorded switch-off
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: medium
updated_date: 2026-10-14 14:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two watches can share one Dreams source. When Dreams is switched off, the source is stopped as soon as any of its watches has recorded the stop; if one watch's record write failed, the stop goes ahead and only the other watch owes the source a restart. If that watch is then untracked while Dreams is off, the debt is left on a retired watch: after switching on, the remaining watch is active but its source stays off, and tracking the story again reports it attached while leaving it off. Only untracking and tracking again repairs it. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After any sequence of failed and successful record writes, switching Dreams back on restarts every source a live watch needs
- [x] #2 Tracking a story whose live watch's source is off turns that source back on
- [x] #3 A test reproduces the two-watch sequence with real stores and fails before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on dev with the reviewer's two-watch probe: A's switch-off write fails, B's lands, B is untracked while Dreams is off; after switching on A is active and its source stays off, and tracking story one again reports attached and leaves it off.
2. Tests first (real stores): the sequence with and without a second switch-off pass, then switching on restarts the source; and tracking the story again restarts a source a live watch needs when the debt is left on a retired watch.
3. Root cause: a retiring watch whose record owes the source a restart keeps that debt on its retired record. Hand the debt to a live watch of the same source when the watch retires (untrack and sweep share _record_retirement); write the heir's record first, and keep the debt on the retired record if that write fails.
4. Re-track (the story's own live watch): when its source is off and Dreams can account for the disable (a record of any watch of that source owes it), restore it like a held page, after the public-address check.
5. Run Tests/Dreams; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced on dev 1ef4fcf4ae with the reviewer's two-watch probe: A's switch-off write fails, B's lands, B is untracked while off; after switching on A is active and its source is off, and tracking story one again reports attached and leaves it off.
- Root cause: a retiring watch whose record owes its source a restart kept that debt on its retired record, which no switch pass reads. _record_retirement (untrack and the sweep) now first hands the debt to a live watch of the same source (_hand_on_source_debt: the heir keeps what it owed and its settled state when that was off, else it is left unsettled so the next switch-on takes it up), and drops it from the retired record only when that landed.
- Re-track: _retry_held_source (the story's own live watch) also restarts a Dreams-created source that is off when Dreams can account for the disable (_dreams_can_account_for_disable over every watch of that source, retired ones included), after the same public-address check. That repairs a profile an earlier build left this way and the case where the hand-over write failed. A source the user switched off stays off.
- Tests: test_a_live_watch_gets_its_source_back_after_its_sibling_left (switching on at once, and after another switch-off pass) and test_tracking_a_story_again_restarts_the_source_its_watch_needs (the hand-over write failing); all three red before the fix (source left off).
- One logger statement added (warning: handing subscription {} on failed, id and exception type only).
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919: Tests/Dreams 868 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
