---
id: TASK-41138
title: "Console and Buddy test leftovers: a two-MCP-rounds payload flake, a cold-home Buddy child deadline, test_buddy_conversation_modal.py on no lane, and test_fleet_wake_retry.py without a profile marker"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- console
- buddy
- tests
priority: low
references:
- Tests/UI/test_console_mcp_approval.py
- Tests/UI/test_buddy_conversation_modal.py
- Tests/Chat/test_fleet_wake_retry.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Each one either fails intermittently on a laned file or leaves behaviour unguarded.

**Evidence.**
- `test_console_mcp_approval.py::test_two_mcp_rounds_for_the_same_session_the_earlier_ones_teardown_does_not_evict_the_newer_ones_payload` (`:3236`) was reported flaky by the round-14 Console stream; the file is a whole UI census entry, so a flake reaches the required check. Not reproduced at closeout.
- `test_buddy_conversation_modal.py::test_cold_home_saved_buddy_bootstraps_only_after_explicit_open` fails with the base modal too: a child process's 5 s `until` deadline under load (TASK-40315's notes, load 62).
- `Tests/UI/test_buddy_conversation_modal.py` is in no workflow and not in the census.
- `Tests/Chat/test_fleet_wake_retry.py`: its three nodes fail the same with the base service; "that file has no profile marker" (TASK-40313's notes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The payload flake is reproduced under load and fixed, or the notes record it could not be reproduced in 30 runs at load 40 or more
- [ ] #2 The cold-home Buddy test's deadline comes from measured times with 2x headroom, or waits on its condition
- [ ] #3 test_fleet_wake_retry.py gets the profile marker its config reads need and its three nodes pass
- [ ] #4 test_buddy_conversation_modal.py's green nodes run on a PR lane
<!-- AC:END -->
