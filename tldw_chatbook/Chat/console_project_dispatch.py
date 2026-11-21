"""Project-instruction dispatch callbacks for one Console agent run.

Moved out of ``ConsoleChatController._run_agent_reply`` (module size ratchet).
Controller-module helpers are looked up at call time through ``host`` so tests
that patch them on ``console_chat_controller`` keep reaching them.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from tldw_chatbook.Chat.console_project_instructions import (
    project_instruction_notice_key,
)

#: Session id -> (assistant message id, the user's own "cancel"/"disable"
#: answer to the primary send's notice), held until that reply is finalized
#: (TASK-33621.98). Keyed to the reply so a record a Stop or an exception left
#: unclaimed can never label another turn; each new run's callbacks clear it.
#: An automatic cancel (the selection drifted, or the chat was already
#: accepted) and a review destination's notice are not recorded.
_DECLINED: dict[str, tuple[str, str]] = {}

_DECLINED_COPY = {
    "cancel": (
        "Not sent: this folder's project instructions were not approved. Retry "
        "to answer again: Proceed sends them, Disable turns them off for this chat."
    ),
    "disable": (
        "Not sent: project instructions are now off for this chat. Retry to "
        "send without them."
    ),
}


def take_declined_notice(session_id: str, assistant_message_id: str) -> str | None:
    """Claim the decline recorded for exactly this reply, if there is one."""
    reply, decision = _DECLINED.pop(session_id, (None, None))
    return decision if reply == assistant_message_id else None


def finalize_declined_notice(
    controller: Any,
    assistant_message_id: str,
    session_id: str,
    run_id: str | None,
    decision: str,
) -> Any:
    """End a turn whose project-instructions notice was declined (TASK-33621.98).

    It used to end like any runtime cancellation: a failed reply reading only
    "Response failed." and ``blocker=-`` in the log. The failed reply now
    carries the "Not sent" copy naming the decline (Retry on it sends the turn
    again) and the turn is BLOCKED, as ``_finalize_agent_cancelled`` shapes it.
    """
    from .console_chat_controller import ConsoleSubmitResult
    from .console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmitBlocker,
    )

    copy = _DECLINED_COPY[decision]
    if controller._ensure_assistant_placeholder(assistant_message_id, session_id):
        controller.store.append_stream_chunk(assistant_message_id, copy)
        failed = controller.store.mark_message_failed(assistant_message_id)
    else:
        failed = controller._append_failed_assistant(session_id, copy)
    controller._record_run_assistant_message(run_id, failed)
    controller._set_run_state(ConsoleRunState.blocked(copy), session_id=session_id)
    return ConsoleSubmitResult(
        True,
        True,
        copy,
        terminal_status=ConsoleRunStatus.BLOCKED,
        blocker=ConsoleSubmitBlocker.PROJECT_INSTRUCTIONS_DECLINED,
    )


def build_project_dispatch_callbacks(
    controller: Any,
    *,
    session_id: str,
    registry: Any,
    project_selection: Any,
    state: Any,
    resolution: Any,
) -> tuple[Any, Any, Any]:
    """Return ``(activation_callback, authority_guard, confirm_dispatch)``.

    Consent is checked against ``state``, captured at preparation; only this
    run's own committed decisions advance it (review destinations confirm
    again within the same run).
    """
    from tldw_chatbook.Chat import console_chat_controller as host

    project_owner_loop = asyncio.get_running_loop()
    expected_project_state = state
    _DECLINED.pop(session_id, None)  # a new run: no unclaimed decline survives

    def on_owning_loop(callback):
        # Retained review authority may run on its captured app loop.
        try:
            if asyncio.get_running_loop() is project_owner_loop:
                return callback()
        except RuntimeError:
            pass
        app = controller.app
        call_from_thread = getattr(app, "call_from_thread", None)
        # Textual only permits call_from_thread from another thread.
        if callable(call_from_thread) and threading.get_ident() != getattr(
            app, "_thread_id", None
        ):
            return call_from_thread(callback)
        return callback()

    def project_activation_callback(event):
        return on_owning_loop(
            lambda: controller._record_project_instruction_activation(session_id, event)
        )

    def project_authority_guard():
        return bool(
            on_owning_loop(
                lambda: host.project_instruction_authority_is_current(
                    store=controller.store,
                    session_id=session_id,
                    registry=registry,
                    expected_selection=project_selection,
                )
            )
        )

    def confirm_project_dispatch(snapshot, target_resolution=None):
        active_resolution = target_resolution or resolution
        active_notice_key = project_instruction_notice_key(
            project_selection.locator_fingerprint,
            active_resolution.execution_key or active_resolution.provider,
            getattr(active_resolution, "base_url", None),
        )

        def commit_and_record(decision):
            nonlocal expected_project_state
            committed = host.commit_project_instruction_dispatch_decision(
                store=controller.store,
                session_id=session_id,
                registry=registry,
                expected_state=expected_project_state,
                expected_selection=project_selection,
                notice_key=active_notice_key,
                decision=decision,
            )
            if committed in {"proceed", "disable"}:
                current = next(
                    (
                        item
                        for item in controller.store.sessions()
                        if item.id == session_id
                    ),
                    None,
                )
                if current is not None:
                    expected_project_state = current.project_instruction_state
            if committed != "prompt":
                if committed == "proceed":
                    controller._remember_project_instruction_delivery(
                        session_id, snapshot
                    )
                else:
                    controller._clear_project_instruction_delivery(session_id)
            return committed

        initial = on_owning_loop(lambda: commit_and_record(None))
        if initial != "prompt":
            return initial
        if controller._chat_start.is_accepted(session_id):
            return "cancel"
        callback = controller._confirm_project_instruction_dispatch
        notice = host.build_project_instruction_dispatch_notice(
            snapshot,
            session_id=session_id,
            resolution=active_resolution,
        )
        decision = callback(notice) if callback is not None else "cancel"
        if decision not in {"proceed", "cancel", "disable"}:
            decision = "cancel"
        committed = on_owning_loop(lambda: commit_and_record(decision))
        reply = controller._active_assistant_message_ids.get(session_id)
        if committed == decision != "proceed" and target_resolution is None and reply:
            _DECLINED[session_id] = (reply, committed)
        return committed

    return (
        project_activation_callback,
        project_authority_guard,
        confirm_project_dispatch,
    )
