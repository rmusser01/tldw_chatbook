---
id: TASK-39161
title: "Redesign test_reattach_render_failure_retries_before_decisions_and_view_timers around a path that still builds a fresh Console view"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_store_continuity.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-39120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39120 on round-12 branch fu/r12-ctests). Checked on the round-12 stack 9091820bf2, 2026-11-08.

- The test needs a fresh Console view to re-attach after navigation: it checks that a failed re-attach render retries before the decision and view timers.
- Since fed0b7257a (TASK-31520, 2026-09-05), the Console route is reused. Navigating back resumes the same screen, so no view starts and no `_restore_collapsible_states` timer is set.
- TASK-39120's notes classify the red as test drift from that commit. They propose a redesign "at a path that still builds a fresh Console (the reusable-route cache dropped on a runtime-identity flip)".
- On the stack, under a temporary `bootstrap_profile` marker (scratch plugin, run.sh, `-n 4`, load ~25), the node fails `assert [] == [True]`. The file's other five nodes pass. Plain, the file stops at `RecoveryRequired`.
- No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test drives a path that builds a fresh Console view (for example a runtime-identity flip that drops the route cache) and still proves its intent: a failed re-attach render retries before the decision and view timers
- [ ] #2 With the retry temporarily removed, the test is red; with it restored, green
- [ ] #3 The node passes 3 of 3 runs at `-n 4`, with load recorded
<!-- AC:END -->
