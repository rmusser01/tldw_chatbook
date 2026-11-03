---
id: TASK-38691
title: "Console command-composer and send-draft snapshot test files run on no PR lane"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- tests
- console
- ci
priority: low
references:
- Tests/UI/test_console_command_composer.py
- Tests/UI/test_console_send_draft_snapshot.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-33621.36
- TASK-37141
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down, which reported that no PR lane runs `Tests/UI/test_console_cost_chip_screen.py`, `Tests/UI/test_console_send_draft_snapshot.py` or `Tests/UI/test_console_command_composer.py`. Checked on dev 9bbd5c2515, 2026-11-03: neither of the last two is in `scripts/ui_pr_gate_census.txt` or a workflow target. TASK-35831 AC#3 already covers the cost-chip file's lane.

One run of both files with `-n 4` gave 57 failed and 77 passed: 52 in the command-composer file and 5 in the snapshot file.

**Command-composer file (52 failed).**
- 48 are `RecoveryRequired: raw_source_selection_changed` (the TASK-38111 trap). This file has only two `bootstrap_profile` marks (:898, :1038) and is not on `Tests/conftest.py`'s bootstrap-profile list. Open TASK-37141 already names this file's setup-profile failures (its AC#1-#3), so it owns them.
- The other four:
  - `test_escaped_chat_stash_keeps_segment_payload_and_restore_consistent`: its stub screen lacks `_ui_responsiveness_monitor`;
  - `test_raw_cli_refused_stash_restore_preserves_exact_trusted_payload`: `assert False is True`;
  - `test_raw_cli_insertions_preserve_trust_only_at_command_body_boundary[2-True-paste]` and `[2-True-file]`: `ValueError: raw CLI command must not be empty or whitespace`.

**Snapshot file (5 failed, none `RecoveryRequired`).** The file is already module-marked `bootstrap_profile` (:40), so the marker half of TASK-33621.36 AC#4 is done. That AC still owns these reds:
- `test_console_submit_exception_retains_exact_recovery_and_keeps_app_alive`: `assert 'precious draft' == ''`;
- `test_console_fresh_profile_first_send_resolves_real_session_not_sentinel`: an awaited `asyncio.Event` is never set;
- `test_console_armed_unknown_mouse_send_snapshots_before_skill_await`: refused with `['Draft changed; Send again.']`, which is the received path's live-vs-stash check (TASK-38689); PR #3376 isolates the test from it;
- `test_console_enter_schedules_visible_send_without_a_watchdog`: `assert None is not None`;
- `test_console_deleted_owner_refuses_and_keeps_captured_inputs`: expected the warning "Console session is closed." and got "coroutine raised StopIteration".

**Boundary.** TASK-37141 owns the command-composer file's profile and setup errors. TASK-33621.36 AC#4 owns the snapshot file's reds. This task owns the command-composer file's remaining reds and puts both files on a lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every red in Tests/UI/test_console_command_composer.py that is not a profile or setup error (the four above, and any that TASK-37141's fix uncovers) passes on dev, or is rewritten or retired with a recorded reason
- [ ] #2 Both files are in the UI PR-lane census, or the task notes record why one cannot be
<!-- AC:END -->
