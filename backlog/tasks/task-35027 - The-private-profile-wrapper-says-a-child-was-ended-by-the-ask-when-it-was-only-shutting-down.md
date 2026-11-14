---
id: TASK-35027
title: The private-profile wrapper says a child was ended by the ask when it was only
  shutting down
status: Done
created_date: 2026-10-15 15:48
labels:
- tests
- ci
updated_date: 2026-10-18 09:40
assignee:
- '@claude'
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a private-profile child is still running at the parent's wait deadline, the parent asks it for its stacks with SIGUSR1 and reports how it ended. If the child dies of the signal's default action, the report says it was ended by the ask before it could answer, which the wrapper attributes to a child still starting up. Python also resets signal handlers while the interpreter shuts down, so a child that had already finished its test and was exiting gets the same wording, as happened in run 38613866408 on 2026-10-14. The report then points a reader at start-up when the real story is a slow exit. Found in the review of PR #3117.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A child that had finished its test and was exiting when asked is reported as exiting, not as starting up
- [x] #2 The start-up case keeps its current wording
- [x] #3 A test covers both cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find a signal in the parent that tells a finished child from one still starting up.
2. Report the finished case as exiting; keep the start-up wording.
3. Pin both with stand-in children.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The child's JUnit report is that signal: pytest writes it when the session ends, and Python puts signal handlers back to their defaults only later, while the interpreter shuts down (checked: atexit runs before that reset, and a SIGUSR1 in the finalizer phase ends the child with -30). So when the ask's default action ended the child (-SIGUSR1) and the report already records a test case, the child had finished and was exiting; Tests/private_profile.py now says 'its test had ended and it was exiting when the ask for its stacks ended it' (ended, not finished: a test that failed, even by its own timeout, has a report too). A hung test or teardown has not written the report, so it cannot be reported as exiting; with no report the start-up wording is unchanged. New helper _child_finished(report) (a report that does not parse counts as not finished).

Tests (Tests/CI/test_private_profile_child_deadline.py): a stand-in child that writes a report, resets SIGUSR1 to the default and keeps running is reported as exiting (fails without the change: it got the start-up wording); the existing start-up test now also asserts it is not called exiting. Tests/CI and the private-profile wrapper tests: 993 passed, 1 skipped.

Files: Tests/private_profile.py, Tests/CI/test_private_profile_child_deadline.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A private-profile child that the parent's stack request ends after its test ended is now reported as exiting, not as starting up; the start-up wording is unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
