---
id: TASK-34985
title: A watch re-tracked after an untrack made while Dreams was off keeps its change
  alert off
status: Done
created_date: 2026-10-14 15:45
labels:
- dreams
- correctness
priority: low
updated_date: 2026-10-14 19:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
If a user untracks a tracked page while Dreams is off, switches Dreams on, and then tracks the same page again, the page's source comes back but its change alert stays off: the new watch reuses the existing alert rule as it stands, and the switch-off had disabled it. The user gets a tracked page that never alerts. Confirmed with a probe on real stores while working the TASK-34919..34928 follow-ups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tracking a page again after this sequence leaves its change alert enabled
- [x] #2 A user's own disabled alert rule on their own source is not re-enabled by it
- [x] #3 A test reproduces the sequence with real stores and fails before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on dev c12611946c: untrack while Dreams is off, switch on, track the page again: the source is back, its change rule stays off.
2. Root cause: the switch-off records the Dreams rule it silenced on the watch's record; retiring that watch while off writes a retirement record with no rules, so nobody owes the rule a re-enable, and a re-track reuses the rule as it stands.
3. Tests first (real stores): the sequence on a Dreams-created source, and on a user's own source with the user's own disabled rule beside Dreams' rule (the user's stays off).
4. Keep the owed rules on the retirement record (and when a re-track settles the source); when tracking attaches to the subscription again, re-enable the Dreams rules a retired watch's record still owes, then drop that debt. Only rules a record owes are touched: a rule the user switched off, or a user's own rule, never is.
5. Run Tests/Dreams; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced on dev c12611946c (the probe from the TASK-34919 follow-ups): after an untrack while Dreams was off, a switch-on and a re-track, the source is back and Dreams' change rule stays off.
- Root cause: the switch-off records the Dreams rule it silenced on each watch's record; _record_retirement wrote the retirement record with no rules, so once the watch retired while Dreams was off nobody owed the rule a re-enable, and the re-track reused the rule as it stood (a rule that exists is shared as it stands).
- Fix (track_service): the retirement record keeps the rules it owed, and so does _settle_source_restored for a retired rider; when track_page attaches to the subscription again and Dreams' rule exists, _restore_rules_left_off re-enables the Dreams rules a retired watch's record still owes and drops that debt from the record. Only rules a record owes are touched: a Dreams rule the user switched off by hand (never recorded by the switch) and a user's own rule (not Dreams', never on a record) stay as they are. Tracking is refused while Dreams is off (TASK-34928), so this only runs with Dreams on.
- Test: test_tracking_again_after_untracking_while_off_alerts_again, on a Dreams-created source and on a user's own source carrying the user's own disabled rule beside Dreams' (that one stays off); both red before (Dreams' rule left off).
- No logger change.
- Verified 2026-10-14 on fix/task-34924-dreams-restore-boot-check off dev c12611946c: Tests/Dreams 897 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
