"""Canvas tool registration for the current, explicitly owned agent turn."""

from __future__ import annotations

from typing import Any

from . import console_chat_controller as host


def register_canvas_turn(
    self: Any,
    *,
    session_id: str,
    conversation_id: str | None,
    assistant_message_id: str,
    session: Any,
) -> tuple[Any, Any]:
    """Advertise Canvas only after the existing enabled and native-scope checks."""
    canvas_provider = None
    canvas_authority = None
    canvas_controller = getattr(self.store, "canvas_turn_controller", None)
    try:
        canvas_enabled = self._canvas_enabled_reader() is True
    except Exception:  # noqa: BLE001 - Canvas tool advertising fails closed
        canvas_enabled = False
    if canvas_enabled and canvas_controller is not None and session is not None:
        from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
        from tldw_chatbook.Canvas.models import CanvasScope

        canvas_run_id = str(host.uuid4())
        canvas_scope = CanvasScope(
            session_id=session_id,
            conversation_id=conversation_id,
            active_message_ids=self.store.canvas_active_path_message_ids(session_id),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id=canvas_run_id,
            ephemeral=session.ephemeral,
        )
        canvas_scope = canvas_controller.capture_selected_scope(canvas_scope)
        canvas_run = canvas_controller.register_run(
            canvas_scope,
            assistant_message_id=assistant_message_id,
            temporary=session.ephemeral,
        )
        canvas_provider = CanvasToolProvider(
            canvas_run,
            scope=canvas_scope,
            enabled_reader=self._canvas_enabled_reader,
            disabled_reader=self._canvas_disabled_reader,
        )
        canvas_authority = canvas_provider.issue_registration_authority()
    return canvas_provider, canvas_authority
