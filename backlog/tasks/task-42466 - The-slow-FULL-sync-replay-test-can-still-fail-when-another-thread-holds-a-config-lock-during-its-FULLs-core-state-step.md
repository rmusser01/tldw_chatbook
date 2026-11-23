---
id: TASK-42466
title: "The slow-FULL-sync replay test can still fail when another thread holds a config lock during its FULL's core-state step"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- console
- tests
- hardening
priority: low
references:
- Tests/UI/test_console_projection_reads_off_loop.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42425. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-42425 traced three Console projection-cadence CI failures to a sync's checked render meeting the storage coordinator while another thread held it, and isolated the tests from that contention (`_isolate_checked_renders`). Its notes leave one exposure unguarded. `Tests/UI/test_console_projection_reads_off_loop.py::test_a_full_sync_slower_than_the_display_life_does_not_replay_itself` (line 261 at the stack top) awaits a FULL sync that runs `sync_console_core_state` (`UI/Console_Modules/config_sync.py:511`). That step probes config locks without blocking. If another thread holds one, it sets the whole-sync replay flag, and the test fails with "a FULL sync whose awaits outlived the display replayed itself" (line 305 at the stack top; line 266 on TASK-42425's base). TASK-42425 saw this only locally, in a background task, never in this node, and never in CI: its 12 CI failures were all the storage coordinator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a temporary edit (reverted) that holds a config lock from another thread during the FULL's core-state step, the test fails with its "replayed itself" message; the red is recorded in the notes
- [ ] #2 The test is isolated from config-lock contention it does not measure, and passes under that forced hold, while still failing if a slow FULL replays itself for its own reasons
- [ ] #3 The file passes in repeated runs under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
