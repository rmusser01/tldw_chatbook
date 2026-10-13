---
id: TASK-34923
title: Make the hand-over of a Dreams event reminder durable
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a watch that holds an event reminder is untracked, the reminder is handed to another live watch of the same event. Four gaps remain: the receiving watch's record is written after the reminder is re-linked, so a failed write leaves the reminder linked to it, disabled and owed by nobody, and switching on never re-arms it; the reminder keeps the other watch's title; a hand-over landing during the switch-off's staging can leave it recorded on nobody; and four parts of the logic are not covered by any test (each survives a mutation): a reminder the user switched off is not handed on armed, a watch that already has its own reminder is not given a second, a passed event is not handed over, and the retired watch's record drops what it handed on. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reminder handed over is re-armed by the next switch-on even when a record write failed during the hand-over
- [ ] #2 A handed-over reminder carries the receiving watch's title
- [ ] #3 A hand-over during a switch-off leaves the reminder recorded on exactly one watch
- [ ] #4 Each of the four named behaviours is covered by a test that fails when it is removed
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
