---
id: TASK-41958
title: "Console Send tests are refused 'Console preparation source unavailable; Send again.' since #3480"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/test_console_send_draft_snapshot.py
- Tests/Chat/test_console_archive_recovery_dispatch.py
- Tests/UI/test_console_send_admission_off_pump.py
dependencies:
- TASK-41633
- TASK-41649
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** 21 nodes in `Tests/UI/test_console_send_draft_snapshot.py` are red on dev; at least 6 are refused "Console preparation source unavailable; Send again." (traced to 11d71e5db5, #3480), 15 untraced. The send/queue covering files `test_console_archive_recovery_dispatch.py` and `test_console_send_admission_off_pump.py` fail the same way, plus `OriginalSourceChanged: original agent storage source changed` and `'ConsoleChatController' object has no attribute '_compose_local_provider_async'`. Node lists: the TASK-41633 notes (F1) and TASK-41649's run logs.

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_blocked_send_retains_exact_recovery_after_runtime_custody`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_custody_snapshots_draft_before_late_keystrokes`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_deleted_owner_refuses_and_keeps_captured_inputs`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_double_enter_sends_once_and_loses_nothing`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_enter_schedules_visible_send_without_a_watchdog`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_fresh_profile_first_send_resolves_real_session_not_sentinel`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_hidden_send_keeps_draft_and_unblocks_next_send`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_mouse_send_cannot_claim_another_sessions_pending_enter`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_pre_durable_failure_keeps_newer_typing_and_recovery`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_same_session_send_waits_for_pending_enter_token`
- `Tests/UI/test_console_send_draft_snapshot.py::test_console_submit_exception_retains_exact_recovery_and_keeps_app_alive`
- `Tests/UI/test_console_send_draft_snapshot.py::test_custody_refusal_recovery_respects_durable_acceptance_and_undo[False-False]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_custody_refusal_recovery_respects_durable_acceptance_and_undo[False-True]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_custody_refusal_recovery_respects_durable_acceptance_and_undo[True-False]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_custody_refusal_recovery_respects_durable_acceptance_and_undo[True-True]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_mouse_send_completion_preserves_text_typed_after_acceptance`
- `Tests/UI/test_console_send_draft_snapshot.py::test_refusal_recovery_restore_never_overwrites_another_visible_owner`
- `Tests/UI/test_console_send_draft_snapshot.py::test_setup_refusal_after_custody_preserves_exact_recovery[False-]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_setup_refusal_after_custody_preserves_exact_recovery[False-new draft]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_setup_refusal_after_custody_preserves_exact_recovery[True-]`
- `Tests/UI/test_console_send_draft_snapshot.py::test_setup_refusal_after_custody_preserves_exact_recovery[True-new draft]`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node group has a recorded cause
- [ ] #2 Drift is repaired and defects fixed red-first; the files pass
<!-- AC:END -->
