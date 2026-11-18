"""Consume scope summaries from the selected backend's retired publication."""

from ...Chat.console_execution_inspector_values import ConsoleInspectorScopes


def _selected(retrieval):
    from .execution_state import selected_client
    from .retrieval import ConsoleRetrievalController

    client = getattr(retrieval.app_instance, "console_execution_client", None)
    if client is None:
        return None
    screen = client._view
    if (
        type(retrieval) is not ConsoleRetrievalController
        or screen is None or screen._retrieval is not retrieval
        or screen.app_instance is not retrieval.app_instance
        or selected_client(screen) is not client
    ):
        raise RuntimeError("Console Inspector scope presentation owner changed")
    return screen, client


def selected_inspector_scopes(retrieval) -> ConsoleInspectorScopes | None:
    """Return complete scope facts only for this exact published chat."""
    selected = _selected(retrieval)
    if selected is None:
        return None
    _screen, client = selected
    value = client.session_composition.inspector_scope
    row = client.projection_view.session(client._projection.session_id)
    if type(value) is not ConsoleInspectorScopes or value.conversation_id != row.persisted_conversation_id:
        raise RuntimeError("Console Inspector scope summary is unavailable")
    return value


async def refresh_selected_inspector_scopes(retrieval, section: str) -> bool:
    """Keep the original explicit-refresh edge after attach/detach completes."""
    selected = _selected(retrieval)
    if selected is None:
        return False
    screen, client = selected
    visit, frame = client._visit, client._projection
    await client.read_projection(
        frame.session_id, frame.incarnation_id, _inspector_refresh=section,
    )
    client.require_view(screen, visit)
    if _selected(retrieval) != selected or (
        client._projection.session_id, client._projection.incarnation_id
    ) != (frame.session_id, frame.incarnation_id):
        raise RuntimeError("Console Inspector scope refresh changed")
    selected_inspector_scopes(retrieval)
    retrieval._request_control_bar_sync()
    return True
