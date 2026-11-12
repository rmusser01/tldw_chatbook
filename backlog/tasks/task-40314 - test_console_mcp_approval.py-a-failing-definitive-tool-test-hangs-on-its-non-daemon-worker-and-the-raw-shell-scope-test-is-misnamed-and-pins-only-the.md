---
id: TASK-40314
title: "test_console_mcp_approval.py: a failing definitive-tool test hangs on its non-daemon worker, and the raw-shell scope test is misnamed and pins only the fallback copy"
status: To Do
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
- [ ] #1 Any assertion failure in the definitive-tool test releases the worker, shuts the controller down and joins the thread in a finally, so the test fails promptly with its own message, and the worker's own failures surface
- [ ] #2 The raw-shell scope test's name matches what it asserts
- [ ] #3 The captured raw-shell session scope copy is pinned by a test
<!-- AC:END -->
