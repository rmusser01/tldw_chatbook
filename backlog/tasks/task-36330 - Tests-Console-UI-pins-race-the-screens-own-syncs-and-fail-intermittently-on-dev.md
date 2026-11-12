---
id: TASK-36330
title: >-
  Tests: Console UI pins race the screen's own syncs and fail intermittently on
  dev
status: To Do
assignee: []
created_date: '2026-10-22 19:15'
labels:
  - tests
  - console
  - flaky
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while re-gating TASK-33620.15.7's branch on origin/dev 7700c63a68 (2026-10-22, a heavily loaded machine, load average 25-43). Each pin below fails intermittently on a detached origin/dev worktree and on that branch alike. All of them mount the real Console through `_mounted_console`, which only waits for the composer to exist, so the screen's own deferred mount work and 0.2 s sync tick can interleave with the test's steps.

- `Tests/UI/test_console_prompt_queue.py::test_full_console_manager_mounts_entry_children_before_live_list_insert` and `::test_navigation_confirmation_is_pure_and_preserves_manager_edit` (which clicks the same Manage button after the same sync; its one failing log was not kept, so the route is likely, not shown): over four full-file runs per tree, dev failed 1 (the manager pin) and the branch 2 (one of each). A diagnostic copy of the file caught the dev failure: right after `await console._sync_native_console_chat_ui()` the queue shelf still held the pre-admit projection (count 0, revision 0, Manage disabled), so `Button.press()` was dropped; one pause later the shelf showed count 1 with Manage enabled, but no manager opened. The sync the test awaits does not always apply the queue projection before returning. Each passes alone (8 of 8 and 6 of 6 per tree).
- `Tests/UI/test_console_composer_reason_width.py::test_chip_going_idle_restores_the_reason_strip_budget`: 2 of 10 runs failed on dev and 1 of 10 on the branch, plus one full-file run on the branch. It sets the voice chip to "error" directly, then asserts the Send disabled-reason strip is hidden; the failing runs find it displayed. The screen's composer action sync (`_sync_console_composer_action_state` -> `composer_run_controls.sync_dictation_presentation` -> `ConsoleComposerBar.sync_dictation_state("idle")`) re-applies the idle chip unconditionally, so a sync landing between the direct `set_voice_status("error")` and the assertion resets the chip.
- `Tests/UI/test_console_provider_continuation_recovery.py::test_real_screen_sync_mounts_updates_and_clears_recovery_callout`: 2 of 3 runs failed on dev and 1 of 3 on the branch, plus one full-file run on dev. After the active assistant id is popped and the screen re-synced, `assert callout.display` finds the recovery callout hidden.

The last two files error at setup under the plain command on this machine (`RecoveryRequired: raw_source_selection_changed`, the TASK-36014 class); they were run with a local plugin that marks every item `bootstrap_profile`, identically on both trees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each pin, what interleaves with the test's steps is traced and named
- [ ] #2 Each pin waits for the event it races (not a fixed sleep), or drives the state through the path the screen itself uses
- [ ] #3 Each pin passes 20 of 20 interleaved runs on dev under load, and still fails when the behaviour it guards is broken
- [ ] #4 The reason-width and provider-continuation files pass under the plain command (with TASK-36014's setup fix or their own bootstrap_profile marking)
<!-- AC:END -->
