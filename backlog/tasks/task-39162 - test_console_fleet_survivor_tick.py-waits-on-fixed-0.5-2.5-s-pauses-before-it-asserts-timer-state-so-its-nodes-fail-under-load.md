---
id: TASK-39162
title: "test_console_fleet_survivor_tick.py waits on fixed 0.5-2.5 s pauses before it asserts timer state, so its nodes fail under load"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_fleet_survivor_tick.py
dependencies:
- TASK-32470
- TASK-39120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39120 on round-12 branch fu/r12-ctests; its notes propose this follow-up). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev's; the file is the same on the stack.

**The fixed pauses.** The file waits a fixed time and then asserts:
- `pilot.pause(0.5)` at `:162` ("let the mount hedge fire against idle"), `:179`, `:222` and `:249`;
- `pilot.pause(2.5)` at `:186` and `:258` (">= 2 survivor-tick beats");
- `pilot.pause(1.5)` at `:226` and `pilot.pause(0.6)` at `:282`.

**What failed.** TASK-39120's runs, on base 8b762b52fa:
- `test_survivor_tick_observes_new_published_usage_without_a_new_timer` is `@private_profile_test`, so it runs plain. It failed at `_console_fleet_survivor_timer is None` after `pilot.pause(0.5)`.
- Under a temporary `bootstrap_profile` marker at load 50-60, `test_the_tick_stops_itself_with_one_final_settle_paint` and `test_survivor_elapsed_advances_with_no_other_interaction` failed their timing preconditions.

**Scope.** TASK-39120 owns the file's seven `RecoveryRequired` nodes (its AC#1); this task is only the waits. TASK-32470 (To Do) also lists this file's survivor-elapsed and unseen-mark tests as harness drift. No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each fixed pause before an assertion is replaced by a bounded wait on the state that assertion reads (the survivor timer, the poll or the paint), which fails naming that state
- [ ] #2 Under a temporary `bootstrap_profile` marker, the file passes 10 of 10 whole-file runs at `-n 4`, with load recorded
- [ ] #3 A temporarily broken wait condition fails on its bounded wait, not on a later assertion
<!-- AC:END -->
