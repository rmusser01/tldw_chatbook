---
id: TASK-38586
title: Guard Watchlists sources pane against phantom row highlight
status: In Progress
created_date: 2026-11-02 14:27
labels:
- bug
- ui
priority: high
updated_date: 2026-11-02 14:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Table rebuilds in sources_pane post a phantom RowHighlighted that retargets the inspector, so Check now or Delete can act on a row the user never chose; every sibling pane already guards with highlight_is_user_driven. See bug-sweep-2026-11-02-dev.md section 1.6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 sources_pane uses the shared highlight_is_user_driven guard or the click-select mixin
- [ ] #2 Typing in the search box no longer retargets the inspector
- [ ] #3 Reproduction test passes
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
