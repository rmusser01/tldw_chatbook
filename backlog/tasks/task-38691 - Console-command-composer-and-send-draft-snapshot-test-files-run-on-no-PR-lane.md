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
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down, which reported that no PR lane runs `Tests/UI/test_console_cost_chip_screen.py`, `Tests/UI/test_console_send_draft_snapshot.py` or `Tests/UI/test_console_command_composer.py`. Checked on dev 9bbd5c2515, 2026-11-03: neither of the last two is in `scripts/ui_pr_gate_census.txt` or a workflow target. TASK-35831 AC#3 already covers the cost-chip file's lane.

One run of both files with `-n 4` gave 57 failed and 77 passed.

**The `RecoveryRequired` trap.** 48 of the 57 failures are `RecoveryRequired: raw_source_selection_changed` (TASK-38111). The files are not on `Tests/conftest.py`'s bootstrap-profile list: the command-composer file has 2 private-profile marks and the snapshot file has 1.

**The other nine:**
- `test_console_command_composer.py`:
  - `test_escaped_chat_stash_keeps_segment_payload_and_restore_consistent`: its stub screen lacks `_ui_responsiveness_monitor`;
  - `test_raw_cli_refused_stash_restore_preserves_exact_trusted_payload`: `assert False is True`;
  - `test_console_collapse_disarms_unknown_command_literal_send`: `LookupError` on the `active_app` ContextVar, plus `ValueError: raw CLI command must not be empty or whitespace`.
- `test_console_send_draft_snapshot.py`:
  - `test_console_submit_exception_retains_exact_recovery_and_keeps_app_alive`: `assert 'precious draft' == ''`;
  - `test_console_armed_unknown_mouse_send_snapshots_before_skill_await`: refused with `['Draft changed; Send again.']`, which is the received path's live-vs-stash check, and PR #3376 isolates the test from it;
  - `test_console_deleted_owner_refuses_and_keeps_captured_inputs`: expected the warning "Console session is closed." and got "coroutine raised StopIteration".

TASK-33621.36 AC#4 owns the snapshot file's plain-command failures and its reds. This task covers the command-composer file's reds and puts both files on a lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in Tests/UI/test_console_command_composer.py passes on dev under the plain pytest command, or is rewritten or retired with a recorded reason
- [ ] #2 Both files are in the UI PR-lane census, or the task notes record why one cannot be
<!-- AC:END -->
