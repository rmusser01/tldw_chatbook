---
id: TASK-2769
title: Two pre-existing Console test failures on dev
status: To Do
assignee: []
created_date: '2026-08-07 06:42'
labels:
  - tech-debt
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while verifying wave 3, both reproduced on clean dev at 22c08f958 in disposable worktrees and therefore NOT wave-3 regressions. (1) Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_restores_draft_when_batch_raises fails deterministically in isolation -- a bare-screen fixture reaching query_one via _sync_console_command_popup. (2) Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_retries_console_follow_after_initial_adapter_failure is a load-dependent pilot.click/pause flake: green in a 10-file slice at both commits, red only in a 30-file run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The generation-actions failure passes in isolation
- [ ] #2 The live-work-handoffs flake is either fixed or its load dependency is documented
<!-- AC:END -->
