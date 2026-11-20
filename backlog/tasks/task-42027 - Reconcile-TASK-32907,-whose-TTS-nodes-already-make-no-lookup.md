---
id: TASK-42027
title: "Reconcile TASK-32907, whose TTS nodes already make no lookup"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:05
labels:
- follow-up
- backlog
priority: low
references:
- Tests/TTS
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41708 (round 17). TASK-32907's two TTS nodes already use MockTransport and make no DNS lookup; under a temporary private profile they pass 2 of 2, so its AC#1 and AC#2 appear to hold.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TASK-32907 is closed with evidence for each criterion, or rescoped with a dated note naming what remains
<!-- AC:END -->

## Implementation Plan

1. Run TASK-32907's two TTS nodes through the wrapper, plain and under a temporary `@private_profile_test` (removed after), with the network guard that now refuses DNS lookups (TASK-41708).
2. Run `Tests/TTS/` whole and `Tests/Subscriptions/` + `Tests/Utils/` through the wrapper with `-n 4` and `--timeout-method=signal`, and record wall time and any timeout.
3. Close TASK-32907 with evidence for each criterion, or rescope it with a dated note naming what remains.

## Implementation Notes

- **Outcome:** TASK-32907 is closed (Done) with evidence for each of its four criteria; its Implementation Notes hold the detail. No code changed.
- **Two TTS nodes, through the wrapper, under the getaddrinfo guard:** plain, 2 failed in 21.6 s on the local `RecoveryRequired: raw_source_selection_changed` trap, no lookup recorded; under a temporary `@private_profile_test` (removed), 2 passed in 4.6 s. Negative control: removing the `MockTransport` replacement (temporary, reverted) made the guard name `socket.getaddrinfo` for both hosts.
- **Suite timings, `-n 4 --timeout-method=signal`, load average 20 to 36:** `Tests/TTS` 563 s, no timeout, no egress. `Tests/Subscriptions` + `Tests/Utils` 1,368 s, no egress; 1,200 s of it is four `test_briefing_cast.py` tests that wait unbounded on an event the code under test never sets once it fails on the local trap. With the file temporarily marked `bootstrap_profile` (reverted) those four pass in 7.7 s.
- **What TASK-32907 got wrong:** the 21 s and the quoted assertion were the local trap and traceback context, not DNS; its 17 to 20 minute "stall" was those four 300 s timeouts. Lesson added to `backlog/docs/lessons-testing-evidence.md`.
- **Proposed follow-up (not filed here):** bound the four `await entered.wait()` calls in `Tests/Subscriptions/test_briefing_cast.py` (lines 865, 929, 1072, 1472) so an early failure of the task under test fails the test at once instead of after 300 s.
- Files: `backlog/tasks/task-32907 - Two-TTS-tests-issue-real-network-calls-and-stall-the-suite.md`, `backlog/docs/lessons-testing-evidence.md`.
