---
id: TASK-38586
title: Guard Watchlists sources pane against phantom row highlight
status: Done
created_date: 2026-11-02 14:27
labels:
- bug
- ui
priority: high
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Table rebuilds in sources_pane post a phantom RowHighlighted that retargets the inspector, so Check now or Delete can act on a row the user never chose; every sibling pane already guards with highlight_is_user_driven. See bug-sweep-2026-11-02-dev.md section 1.6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sources_pane uses the shared highlight_is_user_driven guard or the click-select mixin
- [x] #2 Typing in the search box no longer retargets the inspector
- [x] #3 Reproduction test passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Sibling-idiom highlight_is_user_driven guard on both handlers plus once-per-instance row-0 arrival arming in compose so the default-selection contract survives. Commit a30f1797e5.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
