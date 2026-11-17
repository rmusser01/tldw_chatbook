"""Selected Progress readers use one published native-session count map."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_execution_workspace_values import (
    ConsoleWorkspaceCount,
    ConsoleWorkspaceSessionFacts,
    ConsoleWorkspaceSurface,
)
from tldw_chatbook.UI.Console_Modules import execution_state
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail


def test_selected_progress_reader_and_rail_use_published_counts(monkeypatch):
    def local_runtime_read(*_args, **_kwargs):
        raise AssertionError("Selected Progress reached a local runtime reader")

    screen = SimpleNamespace()
    controller = ConsoleAgentController(
        screen,
        app_instance=SimpleNamespace(),
        chat_store_accessor=local_runtime_read,
        provider_gateway_accessor=local_runtime_read,
        native_tool_calls_enabled_accessor=local_runtime_read,
        current_rail_conversation_id=local_runtime_read,
        current_rail_state_accessor=local_runtime_read,
        chat_controller_accessor=local_runtime_read,
        sync_native_console_chat_ui_accessor=local_runtime_read,
        reveal_agent_detail=local_runtime_read,
    )
    monkeypatch.setattr(controller, "_ensure_console_agent_bridge", local_runtime_read)
    rows = tuple(SimpleNamespace(id=value) for value in ("native", "sibling"))
    surface = ConsoleWorkspaceSurface(
        None, True,
        tuple(ConsoleWorkspaceSessionFacts(
            row.id, "", None, None, None, 0, ConsoleRunMarker.NONE,
        ) for row in rows),
        progress_counts=(
            ConsoleWorkspaceCount("native", 3),
            ConsoleWorkspaceCount("sibling", 7),
        ),
    )
    client = SimpleNamespace(
        workspace_surface=surface,
        projection_view=SimpleNamespace(active_session_id="native", session_rows=rows),
    )
    monkeypatch.setattr(execution_state, "selected_client", lambda _screen: client)
    expected = {"native": 3, "sibling": 7}
    assert controller.progress_state() == (3, expected)

    navigation_refreshes = []
    button = SimpleNamespace(is_attached=True, label="")
    rail = SimpleNamespace(
        _agent_progress_state=controller.progress_state,
        _progress_button=button,
        _progress_counts={},
        _refresh_progress_navigation=lambda: navigation_refreshes.append(True),
    )
    ConsoleLeftRail._sync_progress_count(rail)
    ConsoleLeftRail._sync_progress_count(rail)
    assert button.label == "Progress: 3 queued"
    assert rail._progress_counts == expected
    assert navigation_refreshes == [True]

    client.projection_view.active_session_id = "sibling"
    ConsoleLeftRail._sync_progress_count(rail)
    assert button.label == "Progress: 7 queued"
    client.workspace_surface = replace(surface, progress_counts=())
    ConsoleLeftRail._sync_progress_count(rail)
    assert controller.progress_state() == (0, {})
    assert button.label == "Progress: 0 queued"
    assert navigation_refreshes == [True, True]

    # The selected accessor still refuses a stale session publication; it
    # cannot fall back to a local bridge or preserve a prior count map.
    client.projection_view.session_rows = (SimpleNamespace(id="replacement"),)
    with pytest.raises(RuntimeError, match="workspace publication changed"):
        controller.progress_state()
