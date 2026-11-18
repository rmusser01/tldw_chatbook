---
id: TASK-40312
title: "A close during a native commit reports 'Response stopped.' where every other close path reports 'Session closed.'"
status: Done
assignee: ['@claude']
updated_date: 2026-11-13 02:38
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- copy
priority: low
references:
- tldw_chatbook/Chat/console_native_commit.py
- tldw_chatbook/Chat/console_chat_controller.py
- Tests/Chat/test_console_durable_commit_window.py
dependencies:
- TASK-39155
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream, TASK-39155). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The same user action, closing a chat mid-turn, reports two different outcomes depending on timing. That is the drift TASK-22690 set one name to prevent: `SESSION_CLOSED_COPY = "Session closed."` (`console_chat_controller.py:2392-2394`, "One name for it so the six sites that reported a close cannot drift apart").

**Evidence.** TASK-39155 made a close in progress count as explicit (`console_chat_controller.py:25672`). `console_native_commit.py:488-495` then sets `content = "Response stopped."` for an explicit stop, and the same text goes to the run state and the turn result (`:520-534`, `:563-568`, `:582-597`). The temporary-chat setup path ends "Session closed.". Both are pinned: `Tests/Chat/test_console_durable_commit_window.py:149` ("Response stopped.") and `:229` ("Session closed."). The persisted reply text can stay "Response stopped."; the run state and turn result are the close's report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A close during a native commit ends the turn with SESSION_CLOSED_COPY and a stopped state, with no blocker
- [x] #2 The durable-commit tests pin the same copy for both close windows; one is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Confirm on base 26b2e677dd that `_controller_settle_cancelled_native_commit` still reports "Response stopped." for a close (it does: `console_native_commit.py:488-495`).
2. Change the durable-commit test's close case to expect `SESSION_CLOSED_COPY` in the turn result and the run state; show it red.
3. In the settle path, report `SESSION_CLOSED_COPY` when the session has a committed close (`_session_close_generations`), keeping "Response stopped." as the saved reply text and for a plain Stop.
4. Run the durable-commit tests and every test file that pins "Response stopped."

## Implementation Notes

The premise held on base 26b2e677dd: `_controller_settle_cancelled_native_commit` (`console_native_commit.py:488-495`) set "Response stopped." for every explicit cancellation, and returned it as the turn result and run-state copy for a close too.

The settle path now picks a separate report copy: `SESSION_CLOSED_COPY` when the cancellation is explicit and the session has a committed close (`self._session_close_generations`, never cleared, so it also holds after `finalize_session_close`), else the old content. The report goes into both `ConsoleSubmitResult`s and the run state. The saved reply keeps "Response stopped." (the settle write and `accepted_terminal_state_matches` still use `content`), and a plain Stop or app shutdown still reports "Response stopped.".

Test: `test_a_chat_closed_during_its_native_commit_never_reads_as_an_owner_change[close]` now pins "Session closed." in the result and `(STOPPED, "Session closed.")` in the run state, matching the temporary-chat close case below it. Red before the fix: `assert 'Response stopped.' == 'Session closed.'`; green after (file 7/7, `-n 4`).

Covering run (`-n 4`, load 50-97): 26 Chat files that pin "Response stopped." or exercise the native commit and close paths (`test_console_{accepted_*,close_*,durable_*,first_send_*,native_*,runtime_shutdown,turn_resend,provider_continuation,agent_swap,agent_project_instructions,local_citation_boundary}.py`): 602 passed, 27 failed, 2 errors. The same 27 failing nodes against base code (`console_native_commit.py` restored from 26b2e677dd): 26 failed, 2 errors, 1 passed. The one that passed there, `test_console_runtime_shutdown.py::test_cancelled_session_hook_drain_still_finalizes_exact_close`, failed again on base code on a rerun with the same assertion (`settling.wait(2)` for a hook subprocess under load) and submits no draft, so it never reaches the native commit. No new failures.

No User Guide page quotes the close copy. Files: `tldw_chatbook/Chat/console_native_commit.py`, `Tests/Chat/test_console_durable_commit_window.py`.
