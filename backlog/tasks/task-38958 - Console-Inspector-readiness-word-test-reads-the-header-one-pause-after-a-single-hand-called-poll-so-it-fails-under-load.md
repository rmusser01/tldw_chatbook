---
id: TASK-38958
title: "Console Inspector readiness-word test reads the header one pause after a single hand-called poll, so it fails under load"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- tests
- flaky
- console
priority: medium
references:
- Tests/UI/test_console_switcher_local_probe.py
- tldw_chatbook/UI/Screens/chat_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**The flake.** `Tests/UI/test_console_switcher_local_probe.py::test_the_inspector_setup_row_and_send_hint_read_the_readiness_word` fails in both of its cases, `[refused]` and `[key-rejected]`. The UI PR lane runs the whole function (`scripts/ui_pr_gate_census.txt:305`).
- **CI.** It failed on PR #3405, UI lane shard 18/32, 2026-11-04 09:59Z, with `AssertionError: assert 'Not ready · refused :9099' == 'Ready · not tested'` at `:1123`.
- **Local, through the scratch-profile wrapper:**
  - alone, 3 of 3 passed at load 12-15;
  - whole file with `-n 4`, 4 of 10 runs failed at load 34-64 (`[key-rejected]` in runs 2 and 10, `[refused]` in runs 3 and 4).
- Every failure is the same assertion at `:1123`. The readiness value and the readiness word are right; only the painted header `#workbench-header-status` is stale.

**Likely cause.** The test stops the poll timer, calls `_poll_console_credential_readiness()` once, waits one `pilot.pause()`, then reads the header. The poll can return without running `sync()` at four points:
- the pre-gate (`tldw_chatbook/UI/Screens/chat_screen.py:10779`);
- a missed non-blocking settings or config lock (`:10817`);
- the cold-live-read deferral (`:10819`);
- `run_console_config_sync(..., defer=lambda: None)` (`:10829`).

With the timer stopped, nothing retries. TASK-35574's notes record that this test passed 7 of 10 even with the poll made a no-op, because a queued attach repaint wrote the header. So as written it cannot tell the poll from that repaint.

**Related.** TASK-35574 (To Do) owns `test_a_probe_settling_mid_poll_still_reaches_the_console` in the same file, which uses the same harness and fails the same way. In run 10 above it failed alongside this test. One PR may fix both.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both the [refused] and [key-rejected] cases pass 20 of 20 whole-file -n 4 runs at a recorded load of at least 40
- [ ] #2 The test still fails on its assertion when the poll's refresh is removed (a temporary source edit that returns before sync()), with the queued attach repaint drained or excluded so it cannot pass the test on its own
- [ ] #3 The cause is recorded: which poll exit, or which unflushed repaint, left the header reading 'Ready · not tested'
<!-- AC:END -->
