"""Inline chat approval card: the live Phase-5 MCP batch-approval flow
(task-5).

``set_batch``/``ApprovalDecided`` is the UI half of the Console MCP
tool-call approval round-trip: ``ConsoleChatController.
request_mcp_approvals`` (worker thread) pushes a pending batch into
``ChatScreen.chat_state.task_resume_state.pending_approval`` via
``app.call_from_thread``, which flows down through ``sync_task_resume_state``
-> ``ChatTaskCards.sync_state`` -> this card's ``set_batch``. The user's
decisions travel back up as an ``ApprovalDecided`` message that
``ChatScreen`` forwards to ``ConsoleChatController.resolve_pending_approval``
(UI thread), which releases the worker thread's wait.

Every method here stays synchronous end-to-end: ``ChatScreen.
set_task_resume_state``/``sync_task_resume_state`` and ``ChatTaskCards.
sync_state`` are plain sync calls, so ``set_batch`` cannot ``await``
anything either -- see its own docstring for how row remounts stay
collision-safe without awaiting ``remove_children()``.

(task-914: this card used to also carry a legacy single-approval API --
``set_approval``/``#approval-single-body`` with "Allow once"/"Deny"
buttons -- for the pre-task-649 ``Chat_Window_Enhanced`` composition.
That composition was fully retired in task-649, which deleted its last
caller and its dedicated pinning suite (``Tests/UI/test_chat_approvals_
and_resume.py``) without also removing the now-orphaned widget code;
``set_batch`` is the sole production entry point today.)
"""

from __future__ import annotations

from copy import deepcopy
