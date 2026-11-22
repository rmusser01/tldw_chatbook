---
id: TASK-42342
title: "Closing a Console chat during a prompt mix-in read can fail because the close drain does not wait for the read"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:56
labels:
- follow-up
- console
- bug
priority: high
references:
- tldw_chatbook/Chat/console_prompt_mixin_service.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Backup_Recovery/runtime_producer_lifetime.py
- tldw_chatbook/Chat/console_context_repository.py
- Tests/Chat/test_console_context_presentation_current.py
dependencies:
- TASK-34975
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of the PR #3493 port (TASK-34975, round 20, branch fu/r20-ctxclose). Read at 59cc99e61b (dev 8a55555798 plus the round-20 stack); line numbers are that tree's. Found by reading; not reproduced. The context-presentation fix this builds on (9470ee5528) exists only on the TASK-34975 port until that lands.

**Why it matters.** Closing a chat should end with "Session closed." If a prompt mix-in read's native callback is still running when the close finishes its drains, the close fails with `RuntimeError("A native preparation still owns this session.")`. The chat is then left fenced: it can neither send nor be closed again until the app restarts. TASK-34975's notes (2026-11-11 entry) record the same failure for its sibling, the context-presentation read, reproduced in a Windows UAT close and resume.

**Evidence.**
- Both prompt mix-in reads call `run_preparation_read` with `creator` and `reads` set to their `preparation_owner` and pass no `observers`: the snapshot read at `console_prompt_mixin_service.py:418-438` and the attempt-snapshot read at `:818-833`. `preparation_owner` is the controller (`console_chat_controller.py:21191`). Both reads are session-scoped.
- The runtime's `set_chat_controller` (`console_runtime.py:2255-2260`) calls `observe_preparation_reads`, which attaches only reads that already exist at binding time (`runtime_producer_lifetime.py:122-128`). Later mix-in reads live only in the controller's set.
- The close does wait for the read's issuing task: `session_shutdown_tasks` adds it through `preparation_tasks_for` (`console_chat_controller.py:15010-15017`), and the runtime's close waits for those tasks for up to the 2 s grace (`CONSOLE_SESSION_CLOSE_GRACE_SECONDS`, `console_runtime.py:401`). After the grace it cancels whatever is left without awaiting it, drains ordinary native commits, then drains only its own read set (`:6103-6105`) and calls `finalize_session_close`, which raises while the controller still holds a read for the session (`console_chat_controller.py:14909-14911`).
- Nothing bounds the callback itself. `asyncio.timeout(5.0)` around the mix-in reads does not: `run_preparation_read` shields the executor producer and retires the read only when the callback returns (`runtime_producer_lifetime.py:285-312`). So the failure needs a callback that is still running after the grace and the drains that follow, for example a slow database read.
- On that failure the close leaves `_admission_fenced_sessions` (`console_runtime.py:6027`), `_session_close_generations` (`console_chat_controller.py:14750`) and `_session_close_states` (popped only after the check, `:14914`) set. Admission is then refused with "Console session is closed." (`console_runtime.py:2933-2938`), and a second close is refused too (`console_chat_controller.py:14622-14625`).

TASK-34975 fixed the same gap for the context-presentation read: `console_context_repository.py` now passes `observers=(runtime._preparation_reads,)`. Two fixes were proposed: add the runtime observer at both mix-in sites, or fix the root by also draining the controller's `_preparation_reads` for the closing session in the runtime's close path, which would cover any other controller-owned read with the same gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A session close that starts while either prompt mix-in read (the snapshot read or the attempt-snapshot read) has a native callback still running past the close grace waits for that read to retire, then ends with the normal closed-session outcome instead of "A native preparation still owns this session."
- [ ] #2 A test holds each mix-in read's native callback past the close grace, and is shown red on an assertion before the fix and green after
- [ ] #3 The fix sits in the shared close path, or the implementation notes list every controller-owned `run_preparation_read` caller the runtime does not observe and why each one is safe
- [ ] #4 The new tests run on a PR lane
<!-- AC:END -->
