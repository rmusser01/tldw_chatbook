---
id: TASK-39154
title: "Three Console test files fail under their plain command because they carry no profile marker: test_console_rewind_summarize, test_console_run_state_per_session and test_console_mcp_approval"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- test-infra
- console
priority: low
references:
- Tests/Chat/test_console_rewind_summarize.py
- Tests/Chat/test_console_run_state_per_session.py
- Tests/UI/test_console_mcp_approval.py
- Tests/conftest.py
- Tests/private_profile.py
dependencies:
- TASK-38111
- TASK-39141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39138's covering run, round-12 branch fu/r12-ctlr). Checked on dev b590a15afd, 2026-11-08; the round-12 stack fails the same nodes. All runs: run.sh, `-n 4`, load 24-35.

**The runs.** The ctlr stream named six files.
- Plain, they gave 91 failed and 141 passed.
- A scratch plugin then added the `bootstrap_profile` marker to every node. This was a temporary run-time change and was not committed. Under it they gave 16 failed and 216 passed.

**Red only from the per-test profile redirect.** These 75 nodes fail with `RecoveryRequired: raw_source_selection_changed`, or with a symptom of it, and pass under the marker:
- `Tests/Chat/test_console_rewind_summarize.py`: 58. That is 57 `RecoveryRequired`, plus `test_native_message_id_key_stripped_before_provider` (`assert False is True`). Only three of its tests carry a marker.
- `Tests/Chat/test_console_run_state_per_session.py`: 7. That is 3 `RecoveryRequired`, plus 4 `assert False is True` or `TimeoutError`. The file has no marker.
- `Tests/UI/test_console_mcp_approval.py`: 10, all `RecoveryRequired`. One node in the file is `@private_profile_test`.

**Red on their own assertions.** These fail under the marker, and plain too:
- `test_console_run_state_per_session.py::test_submit_draft_closed_session_id_fails_closed_without_touching_active` (`KeyError: 'Unknown Console chat session'`). TASK-39138 (round 12) fixes the cause; its notes say the node passes on its head under a temporary marker.
- `test_console_mcp_approval.py`, 8 nodes:
  - three CSS pins: `_agentic_terminal.tcss` is missing `.approval-row {`, the `.approval-row-decision {` block's `width: 27;`, and the `.approval-row-full-command` block's `width: 1fr`;
  - `test_chat_screen_forwards_approval_decided_to_controller`: `'Mock' object is not iterable`;
  - `test_refusing_one_call_does_not_get_overwritten_by_approving_another` and `test_a_no_app_round_does_not_record_a_user_denial_on_the_mcp_hook`: a `ToolReviewDecision` where a string verdict is expected;
  - `test_revocation_cannot_split_a_batch_snapshot`: `'ReadGap' object has no attribute 'denial_reasons'`;
  - `test_approved_definitive_tool_stays_mounted_until_real_terminal[crash]`: "approved handler never started".

**The other three named files are not this class.**
- `test_console_edit_resend.py` and `test_console_regenerate_branching.py` already carry a module `bootstrap_profile` marker; their reds are their own and out of scope here.
- `Tests/integration/test_console_chat_fork_flow.py` is listed in TASK-33621.36, which now has a dated note with this round's evidence.

No PR lane runs any of the three files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 75 redirect-only nodes pass under their files' plain command, through the suite's existing profile seams; no test binds config to the real profile, and RecoveryRequired admission is unchanged
- [ ] #2 Each own-assertion red is classified as test drift or a product defect, and fixed or retired with the commit that moved the behaviour named; a product defect gets a test that is red before its fix
- [ ] #3 Each of the three files runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
