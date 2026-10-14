---
id: TASK-34921
title: A Dreams switch-on that stops part-way forgets why a watch was paused
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: low
updated_date: 2026-10-14 14:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A watch held at switch-on (for example because its address did not resolve) records the reason, and the Tracked list shows it as paused with that reason. When a later switch-on pass stops part-way, its unfinished record drops the reason: the row loses its paused label while the source is still off, and the next start runs a lookup it is documented not to make. The closing step of the pass also skips watches whose restore failed, and nothing tests that. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A hold reason survives a switch-on pass that does not finish
- [x] #2 The Tracked row of a held watch shows it as paused through any such pass
- [x] #3 The closing step's handling of a watch whose restore failed is covered by a test that fails if the skip is removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check what #3101 (_unsettle_held, failed riders, T3) already closed: run the hold-reason cases on dev 1ef4fcf4ae.
2. Tests first: a held watch whose next pass cannot reach the watchlists service, or runs without it, keeps its record, its Tracked row's paused reason, and a start with no work; a held watch with work left (an owed reminder) keeps its reason in every write of a pass that stops part-way; the closing step's skip of a watch whose restore failed is pinned by a case where removing it does harm.
3. Carry the recorded hold reason through the pass's write-ahead and trim writes, and in the closing step keep it for a watch whose source the pass did not decide (only a page that passed and was un-settled drops it); fold the T3 special case into that rule.
4. Run Tests/Dreams; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Already done by #3101 (checked on dev 1ef4fcf4ae): a held record settled "on" that owes only its subscription is no longer written ahead (N1), so a pass that stops before deciding its source leaves it as it was; when the page passes, its riders are un-settled before the switch-on (S1, re-track path), including a rider whose own step failed (T1); a failed un-settling keeps the hold (T3).
- Still open on dev, and fixed here: (a) a pass that could not reach the watchlists service, or ran without it, wrote a held watch's record back without its reason in the closing step (held dropped, unsettled: the row lost its paused label and the next start looked the name up); (b) a held record with work left (an owed reminder or rule) was written ahead and trimmed without its reason.
- Fix (track_service._resume_pass): the write-ahead and trim writes carry the reason the record had at pass start; in the closing step a watch whose source this pass did not decide keeps that reason (with settled "on" when it owes nothing else), and only riders of a page that passed the check and were un-settled drop it. The T3 special case is now this rule. The docstring sentence about service failures and the unchanged-record skip comment say so.
- Tests: test_a_pass_that_does_not_finish_keeps_the_hold (service-fails, no-service; red before: held dropped), test_a_hold_with_work_left_keeps_its_reason_through_the_pass (every write keeps dns_failure, ends held and settled on with the reminder restored; red before), test_a_watch_whose_restore_failed_keeps_its_last_record (AC 3: passes on dev; red when the closing step's skip of failed watches is removed, which would list a re-armed reminder as owed again; test_a_rider_whose_first_step_failed_is_un_settled_too from #3101 also fails then).
- No logger change.
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919: Tests/Dreams 872 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
