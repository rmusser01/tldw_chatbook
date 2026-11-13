---
id: TASK-40314
title: "test_console_mcp_approval.py: a failing definitive-tool test hangs on its non-daemon worker, and the raw-shell scope test is misnamed and pins only the fallback copy"
status: Done
assignee: ['@claude']
updated_date: 2026-11-13 02:41
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- approvals
priority: low
references:
- Tests/UI/test_console_mcp_approval.py
- tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py
- tldw_chatbook/Chat/approval_presentation.py
dependencies:
- TASK-39154
- TASK-40097
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream, TASK-39154 and TASK-40097). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's. The file is on the UI census whole (TASK-39154), so it runs on a required PR lane.

**The hang.** `test_approved_definitive_tool_stays_mounted_until_real_terminal` (ee098c54f3) starts a non-daemon `threading.Thread(target=run_tool)` (`:1775`) with no try/finally. Only the first wait's failure branch (`:1786-1789`) calls `begin_shutdown()`. The `entered.wait(2)` assertion (`:1792`) and those at `:1795-1815` raise before `release.set()` (`:1817`), and an assertion inside the thread (`:1771`) is swallowed. With the test's 30 s approval timeout, a red becomes a stall of up to about 30 s that hides its own message and eats lane time.

**The scope test.** `test_raw_shell_row_has_no_generic_scope_static_but_an_mcp_row_does` now asserts that the shell row has exactly one `.approval-row-scope`, so its name says the opposite (review r13c-rev1 renamed four others). It pins only the uncaptured fallback, `DECISION_SCOPE_COPY["deny"]` (`chat_approval_card.py:1582-1588`), because the harness passes no view. The captured shell wording ("Until Chatbook exits: future shell commands in this Console chat...", `approval_presentation.py:312-316`) has no test anywhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Any assertion failure in the definitive-tool test releases the worker, shuts the controller down and joins the thread in a finally, so the test fails promptly with its own message, and the worker's own failures surface
- [x] #2 The raw-shell scope test's name matches what it asserts
- [x] #3 The captured raw-shell session scope copy is pinned by a test
<!-- AC:END -->

## Implementation Plan

1. Re-check both premises on base 26b2e677dd and reproduce the hang with a temporary failing assertion, timing the UI lane's serial command.
2. Wrap everything after the worker starts in try/finally: release the handler, `begin_shutdown()`, join, assert the thread ended, and re-raise the worker's own exception (captured in the thread). Make the thread a daemon so a failed join can never hold interpreter exit.
3. Rename the raw-shell scope test to what it asserts.
4. Pin the captured raw-shell session scope with the production row (`RawShellToolProvider.pending_gate_for`) and payload builder (`_build_approval_payload`); show it red under a temporary source edit.

## Implementation Notes

**The hang was worse than the task said.** Both premises held on base. With a temporary `assert False` before the click, the UI lane's serial command (`-p no:xdist --timeout=180`) printed "1 failed in 18.35s" and then never exited: still alive after 10 min (killed). A faulthandler dump (SIGABRT at 90 s) showed the main thread in `threading._shutdown` and the non-daemon worker in `console_interrupt_rounds.run_round`'s wait (`:1359`). The round's 30 s clock counts only while its card is shown, so once the app exited it never timed out. Under xdist the worker process exits without joining, so the stall shows only on the serial lane, where it would hold the shard until its job cap. `--timeout=180` does not cover it: the session had already ended.

**Fix (test only).** Everything after `worker.start()` is in try/finally: `release.set()`, `controller.begin_shutdown()` (which ends the unanswered round as cancelled), `join(5)`, `assert not worker.is_alive()`, then re-raise the worker's captured exception, so it shows with the body's failure as context. The thread is also `daemon=True`. The two ad hoc shutdown/join branches it replaces are gone. After the fix, the same temporary assertion fails in 29.5 s (wall 48 s, load 60) with its own message. A temporary assertion inside the worker now surfaces as "AssertionError: TEMP-40314 worker red" after the body's "approved handler never started". Before, it was swallowed. Both temporary edits were reverted.

**Scope tests.** The misnamed test is now `test_raw_shell_row_has_one_scope_line_beside_its_session_statement`, and its docstring says it pins the uncaptured fallback. The old name appears only in historical QA artifacts under `Docs/superpowers/qa/2026-09-19-approval-action-ownership/` and in task files, which are records and were not edited. New test `test_captured_raw_shell_session_scope_names_this_console_chat` builds the row with the real provider and the payload with the real builder, then checks the Deny default line and, after choosing "Until Chatbook exits", the captured shell scope. Red under a temporary edit that disabled `scope_copy`'s `console_chat` branch: it read the profile-wide copy ("...every call to this tool in profile Console chat..."). Green after reverting.

The shared helper `_session_scope_line` waits (bounded, 5 s) for the decision Select to mount (`Select.NULL` before then) before choosing. The TASK-40315 Buddy test found that a choice made before mount never reaches the scope line.

Runs: `Tests/UI/test_console_mcp_approval.py` whole file `-n 4`, load 60: 123 passed. The file is already a whole-file UI census entry, so no lane edit.

Files: `Tests/UI/test_console_mcp_approval.py`.
