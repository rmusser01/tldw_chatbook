---
id: TASK-42149
title: "test_a_burst_of_keystrokes_runs_the_debounced_pass_once counts five context syncs under load"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_rail_search_debounce.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the library stream (round 18), which saw it fail 3 of 4 local runs at base with `assert 5 <= 4`. Reproduced at d85e90bc9e: under a temporary `@private_profile_test` marker (the plain test otherwise stops at the local `RecoveryRequired: raw_source_selection_changed` trap), 2 of 4 runs at `-n 4`, load 11-16, failed `assert counter.context_syncs <= 4` with 5 (`Tests/UI/test_console_rail_search_debounce.py:226`).

The test counts context syncs during a fixed `pilot.pause(0.35)` after six keystrokes. Its comment allows the debounced pass, its worker's sync and one sync from the 0.2 s native Console tick. Under load more ticks can land in the window, so the count does not isolate what the test pins: that six keystrokes do not cost six passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test pins that six keystrokes inside the debounce window run one debounced pass, counted independently of how many native ticks land in the window
- [ ] #2 It passes 10 of 10 runs under load (load recorded), and a negative control (no debounce, so each keystroke runs a pass) fails it
<!-- AC:END -->
