---
id: TASK-35027
title: The private-profile wrapper says a child was ended by the ask when it was only
  shutting down
status: To Do
created_date: 2026-10-15 15:48
labels:
- tests
- ci
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a private-profile child is still running at the parent's wait deadline, the parent asks it for its stacks with SIGUSR1 and reports how it ended. If the child dies of the signal's default action, the report says it was ended by the ask before it could answer, which the wrapper attributes to a child still starting up. Python also resets signal handlers while the interpreter shuts down, so a child that had already finished its test and was exiting gets the same wording, as happened in run 38613866408 on 2026-10-14. The report then points a reader at start-up when the real story is a slow exit. Found in the review of PR #3117.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A child that had finished its test and was exiting when asked is reported as exiting, not as starting up
- [ ] #2 The start-up case keeps its current wording
- [ ] #3 A test covers both cases
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
