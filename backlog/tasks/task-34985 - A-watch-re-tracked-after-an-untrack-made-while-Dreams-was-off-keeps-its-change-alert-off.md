---
id: TASK-34985
title: A watch re-tracked after an untrack made while Dreams was off keeps its change
  alert off
status: To Do
created_date: 2026-10-14 15:45
labels:
- dreams
- correctness
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
If a user untracks a tracked page while Dreams is off, switches Dreams on, and then tracks the same page again, the page's source comes back but its change alert stays off: the new watch reuses the existing alert rule as it stands, and the switch-off had disabled it. The user gets a tracked page that never alerts. Confirmed with a probe on real stores while working the TASK-34919..34928 follow-ups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tracking a page again after this sequence leaves its change alert enabled
- [ ] #2 A user's own disabled alert rule on their own source is not re-enabled by it
- [ ] #3 A test reproduces the sequence with real stores and fails before the fix
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
