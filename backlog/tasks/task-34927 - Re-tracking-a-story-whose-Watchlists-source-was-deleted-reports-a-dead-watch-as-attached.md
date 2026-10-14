---
id: TASK-34927
title: Re-tracking a story whose Watchlists source was deleted reports a dead watch
  as attached
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: low
updated_date: 2026-10-14 14:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Dreams notice for a source that stopped working tells the user to delete the source in Watchlists and track the story again. Tracking again finds the story's existing watch and reports it attached, with the deleted source's id, so the watch stays active with nothing behind it. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tracking a story again after its source was deleted creates a working watch with a new source
- [x] #2 A test follows the notice's instructions end to end and fails before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: track a story, delete its source in Watchlists, track the story again: the old watch is reported attached with the deleted source's id and stays active with nothing behind it.
2. Test first (real stores), following the instructions end to end: delete the source, track again; expect a new watch on a new, running source, the old watch retired, one armed reminder for the event.
3. Fix in track_page's reuse branch: when the story's live page watch rides a source that no longer exists, retire it (untrack: its reminder is disarmed and recorded) and go on to the attach-or-create path, which makes a new source and watch.
4. Run Tests/Dreams; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced on dev 1ef4fcf4ae with the reviewer's probe: after deleting the source, tracking the story again reported attached, the same watch, the deleted source's id, no source row.
- Fix (track_service.track_page): the reuse branch first asks whether the story's live page watch still has its source (_source_is_gone: get_source raises KeyError). When it is gone the old watch is retired through untrack (its armed reminder is disarmed and recorded on it), and track_page goes on to the attach-or-create path: a new source, rule and watch, and a new reminder (_ensure_reminder's rule for a reminder Dreams disarmed by retiring its watch). This is what untracking and tracking again already did; a reminder the user rescheduled on the old watch is not carried over.
- Test: test_tracking_again_after_deleting_the_source_makes_a_working_watch follows the instruction end to end (delete the source, track again): outcome created, a new running source, the old watch retired, the new watch on the new source with an enabled change rule, one armed reminder on the new watch. Red before (attached).
- No logger change.
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919: Tests/Dreams 873 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
