"""The selected Inspector consumes captured facts without a local runtime."""

from dataclasses import replace
from types import SimpleNamespace

from Tests.UI.test_console_workspace_controller import (
    _browser_row, _workspace_controller, _workspace_state,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_RUN_MARKER_GLYPHS, ConsoleRunMarker,
)
from tldw_chatbook.Chat.console_execution_workspace_values import (
    ConsoleWorkspaceCount, ConsoleWorkspaceReceipt,
    ConsoleWorkspaceSessionFacts, ConsoleWorkspaceSurface,
)
from tldw_chatbook.UI.Console_Modules import execution_state
from tldw_chatbook.UI.Console_Modules.workspace import PageAttemptState
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.conversation_attention import ConversationAttentionFact


def test_selected_inspector_browser_uses_captured_counts_attention_and_markers(monkeypatch):
    def local_runtime_read(*_args, **_kwargs):
        raise AssertionError("Selected Inspector reached a local runtime reader")

    sessions = tuple(SimpleNamespace(
        id=sid, persisted_conversation_id=cid,
    ) for sid, cid in (("live", "saved"), ("named-live", "named")))
    store = SimpleNamespace(sessions=lambda: sessions, active_session_id="live")
    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
        ),
        get_active_workspace=lambda: SimpleNamespace(workspace_id=DEFAULT_WORKSPACE_ID),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry),
        current_chat_store_accessor=lambda: store,
        current_chat_controller_accessor=local_runtime_read,
        ensure_agent_bridge=local_runtime_read,
        subagent_counts_for_rows=local_runtime_read,
        fleet_unseen_ids_accessor=lambda: frozenset({"saved"}),
        run_marker_with_unseen=local_runtime_read,
    )
    row = replace(
        _browser_row("saved", "Saved", workspace_id=DEFAULT_WORKSPACE_ID),
        native_session_id="live", scope_type="global",
    )
    named = replace(_browser_row("named", "Named"), native_session_id="named-live")
    controller._current_console_browser_rows = lambda *_args, **_kwargs: ((row,), 1, "")
    controller._workspace_page_attempts["workspace-7"] = PageAttemptState(rows=(named,))
    controller._console_rail_body_height = lambda: None
    approval = ConversationAttentionFact("approval", "Approval required")
    surface = ConsoleWorkspaceSurface(
        DEFAULT_WORKSPACE_ID, True,
        (
            ConsoleWorkspaceSessionFacts(
                "live", "", None, None, None, 0, ConsoleRunMarker.NEEDS_APPROVAL,
                (approval,),
            ),
            ConsoleWorkspaceSessionFacts(
                "named-live", "", None, None, None, 0, ConsoleRunMarker.NONE,
            ),
        ),
        progress_counts=(ConsoleWorkspaceCount("live", 3), ConsoleWorkspaceCount("named-live", 4)),
        subagent_counts=(ConsoleWorkspaceCount("saved", 7),),
        subagent_count_ids=("saved",),
        unseen_receipts=(ConsoleWorkspaceReceipt("saved", "live", "failed"),),
    )
    client = SimpleNamespace(_workspace_count_ids=())
    monkeypatch.setattr(execution_state, "selected_workspace_surface", lambda _screen: surface)
    monkeypatch.setattr(execution_state, "selected_client", lambda _screen: client)

    state = controller._with_console_conversation_browser_state(_workspace_state())
    (projected,) = state.conversation_browser.sections[0].rows
    assert projected.subagent_count == 7
    assert projected.progress_count == 3
    assert projected.run_marker == resolve_glyph(CONSOLE_RUN_MARKER_GLYPHS[ConsoleRunMarker.NEEDS_APPROVAL])
    assert approval in projected.attention
    assert ConversationAttentionFact("failed", "Failed") in projected.attention
    assert not any(fact.kind == "outcome_unknown" for fact in projected.attention)
    assert state.workspace_tree[0].conversations[0].progress_count == 4
    assert client._workspace_count_ids == ("saved",)

    # Unobserved row sets retain the original pending badge state; prior-set
    # counts cannot be mistaken for a completed read of newly materialized rows.
    surface = replace(surface, subagent_count_ids=("other",))
    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser.sections[0].rows[0].subagent_count == 0
    assert client._workspace_count_ids == ("saved",)
