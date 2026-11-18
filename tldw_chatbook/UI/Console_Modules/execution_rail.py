"""Preserve selected rail-toggle intent while its display facts refresh."""

from collections import deque
import time

from ...Chat.console_domain_services import _definition
from ..Screens.chat_screen import ChatScreen
from .execution_state import _require_methods, refresh_selected_when_due, selected_client


_APPLY = (("_apply_console_rail_section_toggle", _definition(ChatScreen._apply_console_rail_section_toggle)),)


def _identity(client):
    row = client.projection_view.session(client.projection_view.active_session_id)
    return client, client._visit, row.id, row.incarnation_id


def apply_selected_rail_toggle(screen, section_id: str, *, next_open: bool | None) -> None:
    """Retain each actual input in order; a refresh never invents a toggle."""
    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Selected rail toggle requires its actual client")
    if getattr(screen, "_console_process_content_failed", False):
        return
    pending = getattr(screen, "_console_pending_process_rail_toggles", None)
    if pending is None:
        pending = screen._console_pending_process_rail_toggles = deque()
    pending.append((_identity(client), section_id, next_open))
    replay_selected_rail_toggles(screen, client, client._visit)


def replay_selected_rail_toggles(screen, client, visit: int) -> None:
    """Apply current queued inputs once, after the original positive refresh."""
    pending = getattr(screen, "_console_pending_process_rail_toggles", None)
    if not pending:
        return
    client.require_view(screen, visit)
    if selected_client(screen) is not client:
        raise RuntimeError("Selected rail toggle owner changed")
    if getattr(screen, "_console_process_content_failed", False):
        return
    identity = _identity(client)
    while pending:
        captured, section_id, next_open = pending[0]
        if (captured[0] is not client or captured[1:] != identity[1:]
            or visit != client._visit):
            pending.popleft()
            continue
        if time.monotonic() >= client._readiness_deadline:
            refresh_selected_when_due(screen)
            return
        _require_methods(screen, _APPLY)
        # Consume before the original effects. A failing original preference
        # write or later widget update is not permission to apply it twice.
        pending.popleft()
        screen._apply_console_rail_section_toggle(section_id, next_open=next_open)
