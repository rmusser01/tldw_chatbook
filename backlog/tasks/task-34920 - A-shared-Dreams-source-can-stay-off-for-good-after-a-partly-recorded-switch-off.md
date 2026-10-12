---
id: TASK-34920
title: A shared Dreams source can stay off for good after a partly recorded switch-off
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- correctness
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two watches can share one Dreams source. When Dreams is switched off, the source is stopped as soon as any of its watches has recorded the stop; if one watch's record write failed, the stop goes ahead and only the other watch owes the source a restart. If that watch is then untracked while Dreams is off, the debt is left on a retired watch: after switching on, the remaining watch is active but its source stays off, and tracking the story again reports it attached while leaving it off. Only untracking and tracking again repairs it. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After any sequence of failed and successful record writes, switching Dreams back on restarts every source a live watch needs
- [ ] #2 Tracking a story whose live watch's source is off turns that source back on
- [ ] #3 A test reproduces the two-watch sequence with real stores and fails before the fix
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
