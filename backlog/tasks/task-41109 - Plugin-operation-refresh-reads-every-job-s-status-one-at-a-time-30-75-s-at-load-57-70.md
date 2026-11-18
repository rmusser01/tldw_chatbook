---
id: TASK-41109
title: "Plugin operation refresh reads every job's status one at a time, 30-75 s at load 57-70"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 19:25'
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- performance
priority: low
references:
- tldw_chatbook/UI/Plugins_Modules/operations.py
- tldw_chatbook/Plugins/service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Refresh runs inside the awaited button handler, so the Plugins screen does not settle until it finishes. TASK-40311's notes measured the refresh reading 29 statuses one at a time: 30.8 s and more at load about 70, 73-75 s at load about 57.

**Evidence.** `Plugins_Modules/operations.py:210-231` (`refresh`): after the page read, `for identity, job in tuple(self.jobs.items()): state = await self.service.operation_status(identity)`, for every job known to the view, not just the page. Each call goes through the service (`service.py:445`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One refresh reads all statuses it needs in a single service call, or only for the page shown
- [x] #2 Refresh with 29 jobs is measured before and after at the same load and is at least 5x faster
- [x] #3 The plugin review tests that wait on refresh stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure `PluginOperations.refresh()` with 29 jobs (the refresh-growth setup) before the change, with the load average.
2. Add one batched status read (`PluginService.operation_statuses`, one storage round trip; `PluginCoordinator.lookup_operations`, one recovery pass), make the single-identity reads delegate to it, and make `refresh` call it once. The work stays on the plugin storage thread, off the UI loop.
3. Pin it with a test that counts recovery passes per refresh (red before), and check each batched status equals the single read.
4. Measure after at the same load (interleaved A/B), and run the plugin review tests that wait on refresh.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** A refresh now reads every job's status in one service call. The 29-job refresh took 56-103 s before and takes 3.4-5.6 s after, at the same load (load 56-80 before, 70-75 after).

**Cause.** `PluginService.operation_status` made one or two storage calls per job. For a retained operation, the second call was `coordinator.lookup_operation`, which re-runs `recover()` and re-verifies every retained transition. So N statuses cost N full recovery passes, which is quadratic in the history.

**Changes.**
- `PluginCoordinator.lookup_operations(identities)` answers many identities over one `recover()`. It verifies each retained transition once, computes each request-handle nonce at most once and only when asked, and keeps the old per-identity results: the `recovery_required` short-circuit, the ambiguity `ValueError` and `unavailable_or_expired`. `lookup_operation` delegates to it.
- `PluginService.operation_statuses(identities)` answers in two steps. Session custody (fences and drains) is read under the lock, as before. Root operations and retained lookups are read in one `_call` on the plugin storage thread. `operation_status` delegates to it.
- `PluginOperations.refresh` makes that one call for every job. The work stays on the plugin storage thread, off the UI loop (CLAUDE.md "Workers for operations >100ms"). Every job is still refreshed, not just the page, because the review modal's `_refresh` reads all jobs of the installation.
- `Plugins/README.md` names the batch read.
- No restore or authority logic changed.

**Measurements** (a temporary probe, never committed). It used the `refresh_growth` setup: install, 12 data creations and one 12-root deletion, 29 jobs. Each round timed `refresh()` and then a loop of single `operation_status` calls, which is the old per-job cost:
- Before (old code), refresh: 55.9 s at load 55.8, 95.8 s at load 62.9, 103.1 s at load 80.3.
- After (same process, interleaved): refresh 3.4 / 3.5 / 5.6 s against the per-job loop's 79.0 / 93.6 / 114.9 s, at load 75.0 / 69.9 / 74.5. That is 17-27x faster at equal load, and 10-30x faster than the before run.

**Test.** `Tests/Plugins/test_data_cleanup.py::test_operation_refresh_reads_every_status_over_one_recovery_pass` (plugin-qualification lane, whole file; no census change). It counts `coordinator.recover` calls in one refresh of 8 jobs and expects 2: the page and one status read. It also checks that each batched status equals the single read, and that an unknown identity reads `unavailable_or_expired`. Red on the old code: `assert 9 == 2` (one pass per job plus the page), at load ~70. Green after.

**Covering run.** `Tests/UI/test_plugin_review.py`, `Tests/UI/test_plugin_operations.py`, `Tests/Plugins/test_data_cleanup.py`, `Tests/Plugins/test_retention.py` and `Tests/Chat/test_console_plugin_tests.py` at `-n 4`, load 77-128: 169 passed, 2 failed. The refresh tests passed: `test_refresh_growth_keeps_focused_action_and_cleaned_status_painted` (both sizes, 152-179 s at load ~100) and `test_status_refresh_sees_new_retained_root`. The 2 failures are `test_candidate_reload_requires_fresh_discard_review[next]` and `test_installed_manager_collects_all_51_exact_root_refs`. Both hit the 300 s timeout, and neither calls refresh or a status read (the candidates listing and `data_status`). Both failed the same way on dev base cc6006fcf5 code at load 103-116 (307 s, 290 s), so they are pre-existing load timeouts.

**Files.** `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Plugins/service.py`, `tldw_chatbook/UI/Plugins_Modules/operations.py`, `tldw_chatbook/Plugins/README.md`, `Tests/Plugins/test_data_cleanup.py`.
<!-- SECTION:NOTES:END -->
