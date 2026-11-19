---
id: TASK-34928
title: Dreams tracking writes and switch checks that do more than they need
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- performance
priority: low
updated_date: 2026-10-14 15:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small items. Tracking a page or a question does not check the Dreams master switch: the Library hides the action when Dreams is off, but a call that reaches the service anyway re-enables a source and arms a reminder while Dreams is off. And the verifying backfill rewrites the provenance mark of every source Dreams ever created on every run, one write transaction each, even when nothing changed. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tracking while Dreams is off creates or changes nothing and says why
- [x] #2 A verifying backfill that finds nothing to mark makes no writes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: with Dreams off, tracking a second story of a watched page re-enables its source and arms a reminder; the verifying backfill calls set_egress_marked (a write transaction) for every Dreams source on every run.
2. Tests first: tracking a page or a question while Dreams is off raises a reason-coded refusal before any write (nothing created, an existing source stays off, no reminder armed); the story modal says why; a verifying backfill that finds every source marked makes no write in either store.
3. track_page and track_question check the master switch first (TrackDreamsOff, reason_code track_dreams_off); the modal maps it to a notice that names where to switch Dreams on. Tests that tracked with the shipped default (off) set Dreams on.
4. backfill_egress_marks(verify=True) reads the still-to-do list once and writes the marker only for a source it stamped or one on that list.
5. Run Tests/Dreams and the modal tests; guide line; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced on dev 1ef4fcf4ae with the reviewer's probe: with Dreams off, tracking a second story of a watched page re-enabled its source and armed a reminder (outcome re-enabled); a verifying backfill called set_egress_marked once per Dreams source on every run although nothing changed.
- Switch check: TrackDreamsOff (reason_code track_dreams_off) is raised first by track_page and track_question while [dreams] enabled is off, before anything is read or written. The story modal maps it to a notice that says Dreams is off and where to switch it on (Settings > Domain Defaults > Dreams), for Track (t) and Watch question (w). The guide's Tracking section says tracking needs Dreams on.
- Backfill: backfill_egress_marks reads the still-to-do list once; with verify=True it writes the marker only for a source it stamped or one on that list, so a run that finds every source marked writes nothing in either store.
- Tests: test_tracking_while_dreams_is_off_changes_nothing_and_says_why (a second story of a watched page, a new page, a question; nothing created, the source stays off, reminders and records unchanged; red with the check removed: DID NOT RAISE, and the reviewer's probe showed re-enabled), test_a_verifying_backfill_with_nothing_to_mark_writes_nothing (red before: marker 1, marker 2), Tests/UI/test_artifacts_dreams_modal.py::test_tracking_while_dreams_is_off_is_a_notice_and_writes_nothing[track, watch] (red with the modal mapping removed).
- Test changes: the modal track tests patched the Dreams settings to their shipped defaults, which means Dreams off; they now set Dreams on (two cap tests also did, and the page cap test had started passing for the wrong reason). test_track_row_routing_keeps_dream_row_clicks_working failed once under -n 6 and passed alone and in two more full runs of both files (looks like a load flake, unrelated).
- No logger change.
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919: Tests/Dreams + the two modal files + test_library_dreams_reader 934 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
