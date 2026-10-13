---
id: TASK-34919
title: Switching Dreams on can restart the source of a watch retired during the switch
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Dreams is switched on it restores, watch by watch, what the switch-off stopped. The only check that a watch is still live happens at the start of the pass; the watch's source is switched back on later, after every other watch's work and every name lookup. A watch untracked in that window (or retired by the sweep) keeps its source running with its change alert enabled and no live watch behind it, and a later switch-off does not stop it, because the passes read live watches only. Watchlists has no off control, so the user can only delete the source. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A watch retired while a switch-on pass runs ends with its Dreams source and alert rule off
- [ ] #2 A later switch-off or switch-on leaves such a source off
- [ ] #3 A test retires a watch during the pass's name lookup, with real stores, and fails before the fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
