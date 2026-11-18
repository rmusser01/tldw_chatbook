---
id: TASK-34919
title: Switching Dreams on can restart the source of a watch retired during the switch
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: medium
updated_date: 2026-10-14 14:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Dreams is switched on it restores, watch by watch, what the switch-off stopped. The only check that a watch is still live happens at the start of the pass; the watch's source is switched back on later, after every other watch's work and every name lookup. A watch untracked in that window (or retired by the sweep) keeps its source running with its change alert enabled and no live watch behind it, and a later switch-off does not stop it, because the passes read live watches only. Watchlists has no off control, so the user can only delete the source. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A watch retired while a switch-on pass runs ends with its Dreams source and alert rule off
- [x] #2 A later switch-off or switch-on leaves such a source off
- [x] #3 A test retires a watch during the pass's name lookup, with real stores, and fails before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on dev 1ef4fcf4ae with the reviewer's probe (untrack during the switch-on pass's name lookup): source left running, its change rule enabled, no live watch.
2. Test first (real stores): retire the watch during the lookup; after the pass the source and the rule are off, and a later switch-off and switch-on leave them off.
3. Fix in the resume pass: after the per-subscription decisions, for each subscription the pass switched on or whose Dreams rules it re-enabled, read its riders again; when none is live (active or paused) any more, switch the source back off and silence the rules the pass re-enabled.
4. Run Tests/Dreams; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced on dev 1ef4fcf4ae with the reviewer's probe (an untrack landing during the switch-on pass's name lookup): status retired, source active, its change rule enabled, no live watch; a later switch-off and switch-on left both on.
- Fix (track_service._resume_pass): the pass remembers the Dreams rules it re-enabled, by subscription. After the per-subscription decisions, every subscription it switched on or re-armed a rule on is read again (its riders, any status); with no active or paused watch left, the rules it re-enabled are silenced again and the source is switched back off (_disable_dream_source). Retirements in any window of the pass (a sweep, an untrack between the fresh read and the restore, or during the lookup) end the same way. The retired watch's record already says Dreams stopped the source (its write-ahead owed it), so tracking the story again can still switch it back on.
- Test: test_a_watch_retired_during_the_switch_on_ends_with_its_source_off (real stores; the untrack runs inside the patched name lookup); red before the fix (source active=1); it also switches Dreams off and on again and checks both stay off.
- Not changed: a reminder the pass re-armed for a watch retired between the pass's fresh read and the reminder step stays as the retirement left it (untrack and the sweep disarm armed reminders themselves); a rule silenced for a retired watch is shared as it stands by a later re-track of the same page, as after an untrack while Dreams is off.
- One logger statement added (warning: re-checking subscription {} failed, id and exception type only).
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919 off dev 1ef4fcf4ae: Tests/Dreams 865 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
