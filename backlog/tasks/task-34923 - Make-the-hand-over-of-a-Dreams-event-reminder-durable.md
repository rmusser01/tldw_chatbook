---
id: TASK-34923
title: Make the hand-over of a Dreams event reminder durable
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: low
updated_date: 2026-10-14 15:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a watch that holds an event reminder is untracked, the reminder is handed to another live watch of the same event. Four gaps remain: the receiving watch's record is written after the reminder is re-linked, so a failed write leaves the reminder linked to it, disabled and owed by nobody, and switching on never re-arms it; the reminder keeps the other watch's title; a hand-over landing during the switch-off's staging can leave it recorded on nobody; and four parts of the logic are not covered by any test (each survives a mutation): a reminder the user switched off is not handed on armed, a watch that already has its own reminder is not given a second, a passed event is not handed over, and the retired watch's record drops what it handed on. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reminder handed over is re-armed by the next switch-on even when a record write failed during the hand-over
- [x] #2 A handed-over reminder carries the receiving watch's title
- [x] #3 A hand-over during a switch-off leaves the reminder recorded on exactly one watch
- [x] #4 Each of the four named behaviours is covered by a test that fails when it is removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the hand-over (_retire_reminders_sync), the reviewer's surviving mutants (hand-over-even-when-event-passed, handed-still-listed-on-retired, heir-with-own-reminder-still-inherits, user-disabled-handed-on) and the A3c probe (the heir's record write failing).
2. Design: (a) a switch-on completes any hand-over that did not happen: a live watch with an event date takes over the reminder of its story and event that a retired watch's record still owes (not for an event that passed, not when a live watch has one), re-linked and re-titled, written into its own record before it moves, then re-armed; (b) while Dreams is off nothing is handed on at retirement: an armed reminder is disarmed and an owed one stays owed, both on the retired watch's record (one watch), and the switch-on hands them on; (c) while Dreams is on, the heir's record is written before the reminder moves, and if that write fails the reminder stays owed on the retired watch for (a); (d) a handed-over reminder takes the receiving watch's title.
3. Tests first, each seen red on the current code: the failed record write (Dreams off and on), the title, a hand-over during a switch-off pass, and one test per surviving mutant, each checked against its mutant.
4. Run Tests/Dreams; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Design: the switch-on finishes every hand-over that did not happen at retirement. _orphaned_reminders (read-only) finds, for a live watch with an event date, the reminder of its story and event that a retired watch of the same story and event still owes (one Dreams disabled), when no live watch of the story has one -- never for an event that is over (event_passed), one that fired, or one the user switched off (no record owes those). The resume pass writes it into the watch's record first, then re-links it under the watch's title (_take_over_reminders), then re-arms it with the rest.
- At retirement (_retire_reminders_sync): decisions first, writes after. While Dreams is on, the heir's record takes the debt of a switch-disabled reminder BEFORE the reminder is re-linked; if that write fails, the reminder stays owed on the retiring watch's record (the switch-on above finishes it) instead of ending disabled and owed by nobody. While Dreams is off (dreams_setting read by _retire_linked_reminders) nothing is handed on at all: an armed reminder is disarmed and an owed one stays owed, both on the retiring watch's record, so a switch-off pass writing the heir's record meanwhile cannot drop it; the switch-on hands it on. A handed reminder takes the heir's title (_reminder_title, now shared with _ensure_reminder).
- Tests (all red on the committed code before this change): test_a_reminder_handed_on_while_off_is_armed_by_the_next_switch_on (Dreams truly off; retirement record write failing or not), test_a_failed_hand_over_while_dreams_is_on_is_finished_by_the_switch (the heir's record write fails), test_a_handed_over_reminder_takes_the_receiving_watchs_title, test_a_hand_over_during_a_switch_off_leaves_the_reminder_on_one_watch (untrack injected into the pass's stop step).
- AC 4, one test per surviving mutant from the reviewer's mut5/mut6 runs, each seen killed by applying that mutant to the new code: user-disabled-handed-on -> test_a_reminder_the_user_switched_off_stays_off_through_the_switch; heir-with-own-reminder-still-inherits -> test_a_watch_with_its_own_reminder_is_not_given_a_second; hand-over-even-when-event-passed -> test_the_reminder_of_a_passed_event_is_not_handed_over; handed-still-listed-on-retired -> test_the_retired_watch_does_not_claim_the_reminder_it_handed_on.
- Guide: the event reminder paragraph says the reminder takes the remaining watch's name, and moves when Dreams is switched on if the hand-over happened while it was off.
- Note: test_untracking_the_holder_while_dreams_is_off_hands_the_debt_on keeps passing because its settings fixture leaves [dreams] enabled on (only the switch pass runs); it exercises the Dreams-on debt path.
- One logger statement added (warning: Dreams reminder hand-over to item {} failed, id and exception type only).
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919: Tests/Dreams 884 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
