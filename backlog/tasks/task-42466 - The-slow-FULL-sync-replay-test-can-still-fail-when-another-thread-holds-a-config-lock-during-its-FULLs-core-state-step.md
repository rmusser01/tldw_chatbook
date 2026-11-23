---
id: TASK-42466
title: "The slow-FULL-sync replay test can still fail when another thread holds a config lock during its FULL's core-state step"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 21:47
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
- [x] #1 With a temporary edit (reverted) that holds a config lock from another thread during the FULL's core-state step, the test fails with its "replayed itself" message; the red is recorded in the notes
- [x] #2 The test is isolated from config-lock contention it does not measure, and passes under that forced hold, while still failing if a slow FULL replays itself for its own reasons
- [x] #3 The file passes in repeated runs under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: a temporary edit (reverted) where a thread holds a config lock while the FULL's `sync_console_core_state` probes it; the unchanged test must fail with "replayed itself".
2. Fix in the test, following TASK-42425's `_isolate_checked_renders`: a helper makes the core-state push in the test's own task wait for both config locks (bounded) and hold them, so another thread's hold delays the push instead of refusing it. Pushes in other tasks, and a cold config's deferral, are unchanged.
3. Show the fixed test passes under the same forced hold, and still fails "replayed itself" under a temporary mutation that makes the FULL replay for its own reasons.
4. Soak the file at least 20 times at -n 4 under load, plus serial runs with the UI lane's flags; preflight and ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Red (AC#1).** A temporary edit (reverted) added a wrapper around `config_sync.sync_console_core_state` that, on its first call, started a thread holding `config._config_file_lock()` for 0.3 s and waited until the thread had it, then ran the push. Installed right after the test's `_isolate_checked_renders`, it made the unchanged test fail with "a FULL sync whose awaits outlived the display replayed itself" (line 331 of the edited copy), and the hold was logged as fired. The push probes both config locks with `acquire(blocking=False)` and, on a busy one, hands the whole sync to its replay (TASK-36511), which is what the test reads as a slow FULL replaying itself.

**Fix (AC#2, test only).** New helper `_isolate_config_locks(monkeypatch)` beside `_isolate_checked_renders`, following its precedent (TASK-42425). It wraps `config_sync.sync_console_core_state` (the FULL calls it through the module, `chat_screen.py:17660`): a push in the calling task first takes both config locks in the push's own order, waiting up to 30 s, and holds them across the real push. Both locks are reentrant, so the push's own non-blocking probe then succeeds. Pushes in other tasks are unchanged, and a cold config still defers to the replay. The slow-FULL test calls it next to `_isolate_checked_renders`; the monkeypatch also covers the test's second FULL. No product change: the push's refusal under contention is the designed behaviour, and the test does not measure it.
- Under the same forced hold, the fixed test passed 3 of 3 (hold fired each time).
- Still failing for its own reasons, each under the forced hold, with a temporary product mutation (reverted): `rewarm_console_presentation` returning at once, and the core-state push always taking the cold-config deferral. Both fail with "replayed itself".

**Soak (AC#3).** The fixed file, beside other agents' load (1-minute load average 41-79): 20 of 20 whole-file runs at `-n 4` passed (29 of 29 each, 122-208 s), as two loops of 10 that overlapped each other and the agent-rail soak; and 2 of 2 serial runs with the UI lane's flags (`-p no:xdist --timeout=180`) passed, 397 s and 468 s (load 48-55). `ruff check` passes on the file, and the census entry is unchanged (no new test).

**Files.** `Tests/UI/test_console_projection_reads_off_loop.py`.
<!-- SECTION:NOTES:END -->
