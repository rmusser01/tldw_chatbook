"""Disposable Console worktree projection and recovery launch adapter."""

from dataclasses import replace
from typing import Any


def project(screen: Any, payload: dict[str, Any] | None) -> None:
    """Project volatile confirmation without writing resume metadata."""
    screen.set_task_resume_state(
        replace(screen._task_resume_state, pending_worktree_merge=payload)
    )


async def open_recovery(
    screen: Any, *, session_id: str | None = None, after_run_id: str | None = None
) -> None:
    """List owning-session records; a row action enters the retained confirm flow."""
    from tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog import (
        WorktreeRecoveryDialog,
    )

    runtime = screen._console_runtime()
    controller = runtime.chat_controller
    if controller is None:
        screen.notify("Open a Console conversation first.")
        return
    session_id = session_id or controller.store.active_session_id
    if not session_id:
        return
    try:
        helper = runtime.worktree_recovery
        page = await helper.list_work(session_id, after_run_id)
    except Exception:  # noqa: BLE001 - unavailable recovery stays in the owning view
        screen.notify("Agent work recovery is unavailable for this conversation.")
        return
    if controller.store.active_session_id != session_id or runtime.view is not screen:
        return

    def selected(choice):
        if choice is None:
            return
        run_id, action = choice
        if action == "next":
            screen.run_worker(
                open_recovery(screen, session_id=session_id, after_run_id=run_id)
            )
        else:
            # The helper retains the physical task even if this view worker dies.
            conversation_id = page.conversation_id

            async def recover():
                result = await helper.start(session_id, run_id, action)
                if (
                    runtime.view is screen
                    and controller.store.active_session_id == session_id
                    and (
                        current := controller.capture_worktree_recovery_intent(
                            session_id
                        )
                    )
                    is not None
                    and current.persisted_conversation_id == conversation_id
                ):
                    screen.notify(result.message)

            screen.run_worker(recover())

    screen.app.push_screen(
        WorktreeRecoveryDialog(page, busy=session_id in helper.operations), selected
    )
