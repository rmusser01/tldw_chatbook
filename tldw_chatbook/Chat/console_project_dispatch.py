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
        return on_owning_loop(lambda: commit_and_record(decision))

    return (
        project_activation_callback,
        project_authority_guard,
        confirm_project_dispatch,
    )
