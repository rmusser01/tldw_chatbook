---
id: TASK-36520
title: Fresh-mount Console tests wait on the attach-reconcile flag alone, and a resume test relies on fixed pauses
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/pump_probe.py
- Tests/UI/test_console_cost_chip_screen.py
- Tests/UI/test_console_send_draft_snapshot.py
- Tests/UI/test_console_approval_compact_layout.py
- Tests/UI/test_console_command_composer.py
dependencies:
- TASK-36389
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36389's review (PR #3233, 2026-10-23). TASK-36389 audited tests that reopen a cached Console. It added `console_attach_ended(screen)` to `Tests/UI/pump_probe.py`, which waits for the reconcile to end and for the Console's queue to drain. Tests on a fresh mount, and one resume test, were outside its scope.

These wait only for `_console_attach_reconciled` before they watch or assert attach-time work (dev 5b5920b8e2):
- `test_console_cost_chip_screen.py:1174-1180` polls the flag, then wraps `_sync_console_settings_summary` and `_sync_console_cost_chip` in counting mocks. Its own comment says the attach's resume refresh must land first "or it races into the watched window".
- `test_console_send_draft_snapshot.py:355-359` polls the flag before it arms the send it measures.
- `test_console_approval_compact_layout.py:35-37`: `_pending_card` asserts the flag before it injects a synthetic approval.

The reconcile runs `on_screen_resume` and then sets the flag (chat_screen.py:15360-15371). `on_screen_resume` queues more work with `call_after_refresh`, such as the hook refresh worker (:22731-22737), so that work can still be pending when the flag reads True.

`test_console_command_composer.py::test_console_resume_triggered_prompt_insert_survives_stale_session_switch` (:1859) relies on fixed sleeps:
- `pilot.pause(0.1)` (:1886) is meant to let the mount-time sync settle the draft tracker.
- `pilot.pause(0.25)` (:1904) is described as "past the 0.15s consumption timer".

Of the four files, only `test_console_approval_compact_layout.py` is in the UI census.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each named test waits for the attach to end and the Console's queue to drain before the work it watches or asserts, or the task notes record why it cannot race
- [ ] #2 The prompt-insert test waits for the states it asserts instead of fixed pauses
- [ ] #3 A negative control that delays the reconcile's start, as TASK-36389's probes did, fails each old form and passes each new one
<!-- AC:END -->
