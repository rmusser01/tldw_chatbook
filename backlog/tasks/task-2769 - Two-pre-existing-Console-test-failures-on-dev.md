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

## Addendum (wave-4 task 1, 2026-08-07)

A **third** pre-existing failure in the same file, distinct from the two above:
`Tests/UI/test_console_live_work_handoffs.py`'s `_bare_console_screen_for_restore()`
builds a screen with no app, so the tests using it fail on dev. Same
hand-built-screen-fixture class this programme has shipped once per wave.
Also confirmed pre-existing:
`Tests/UI/test_console_dictation_streaming.py::test_the_transcribing_indication_reverts_on_a_mid_capture_stop`
is a timing flake (failed 1 of 4 isolated runs on a clean baseline worktree,
passes in a full-file run).
