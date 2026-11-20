---
id: TASK-39071
title: >-
  Console: the agent-rail stale-probe test fails when an idle settings-summary
  sync publishes the Agent section in its window
status: To Do
assignee: []
created_date: '2026-11-06 16:11'
labels:
  - console
  - tests
  - flaky
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_agent_rail.py::test_full_log_slow_probe_is_off_thread_and_rejects_stale_target replaces the screen's _sync_console_agent_section with a recorder and asserts that nothing published between the stale probe's release and the workers' completion (published == []). An idle settings-summary sync also calls the Agent section sync (the credential-readiness poll or a native UI sync, through the config sync's render and _apply_console_settings_summary_state), with the probe's cache still False, so the test fails whenever one lands in that window. Measured 2026-11-06 while rebasing TASK-36571.1, host load 66-119 from other agents' load generators, runs interleaved: dev 2684e5453f failed 5 of 8 runs and TASK-36571.1's branch 7 of 8, each with assert [False] == [] or [False, False] == []. A stack probe on both trees (2026-11-05, dev 29592ecc9b and the branch) named config_sync.render -> _sync_console_settings_summary -> _apply_console_settings_summary_state as the publisher. Over 20 s of idle the settings summary synced 0-5 times per run on both trees, only from those two entry points (head 14, dev 12 in 5 runs each). The test runs in the CI Tests/UI shards, so it can fail any PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test fails when the stale probe's result (run-a's availability) is published after the target moved to run-b, and passes when only unrelated Agent-section syncs run in that window
- [ ] #2 A mutant that publishes the stale probe's result still fails the test
- [ ] #3 The test passes 10 of 10 runs on dev under concurrent load (for example two other pytest processes running)
<!-- AC:END -->
