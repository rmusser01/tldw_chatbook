---
id: TASK-42342
title: "Closing a Console chat during a prompt mix-in read can fail because the close drain does not wait for the read"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
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
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of the PR #3493 port (TASK-34975, round 20, branch fu/r20-ctxclose). Read at 59cc99e61b (dev 8a55555798 plus the round-20 stack); line numbers are that tree's. Found by reading; not reproduced.

**Why it matters.** Closing a chat should end with "Session closed." A close that lands while a prompt mix-in read is still running can instead fail with `RuntimeError("A native preparation still owns this session.")`, so the chat does not close.

**Evidence.**
- Both prompt mix-in reads call `run_preparation_read` with `creator` and `reads` set to their `preparation_owner`, and pass no `observers`: the snapshot read at `console_prompt_mixin_service.py:418-438` and the attempt-snapshot read at `:818-833`. `preparation_owner` is the controller (`console_chat_controller.py:21191`). Both run under `asyncio.timeout(5.0)` and are session-scoped.
- The runtime's `set_chat_controller` (`console_runtime.py:2255-2260`) calls `observe_preparation_reads`, which attaches only the reads that already exist at binding time (`runtime_producer_lifetime.py:122-128`).
- The runtime's session close waits up to 2 s for the session's tasks (`CONSOLE_SESSION_CLOSE_GRACE_SECONDS`, `:401`), cancels the rest without awaiting them, then drains only its own read set (`console_runtime.py:6103-6105`). A mix-in read may run for up to 5 s. One still in flight when the close reaches `finalize_session_close` is still in the controller's set, and that method raises (`console_chat_controller.py:14909-14911`).

TASK-34975 fixed the same gap for the context-presentation read (9470ee5528): `console_context_repository.py` now passes `observers=(runtime._preparation_reads,)`. Two fixes were proposed: add the runtime observer at both mix-in sites, or fix the root by also draining the controller's `_preparation_reads` for the closing session in the runtime's close path, which would cover any other controller-owned read with the same gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A session close that starts while either prompt mix-in read (the snapshot read or the attempt-snapshot read) is held waits for that read to retire, then ends with the normal closed-session outcome instead of "A native preparation still owns this session."
- [ ] #2 A test holds each mix-in read across a close and is shown red on an assertion before the fix, green after
- [ ] #3 The fix sits in the shared close path, or the implementation notes list every controller-owned `run_preparation_read` caller the runtime does not observe and why each one is safe
- [ ] #4 The new tests run on a PR lane
<!-- AC:END -->
