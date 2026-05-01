from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Input, Select

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.tabs.tab_state_manager import TabStateManager
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, MessageData, TabState
from tldw_chatbook.Widgets.Chat_Widgets.chat_shell_bar import ChatShellBar
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer


class TestChatSessionDataSerialization:
    def test_chat_session_data_round_trip_preserves_runtime_discovery_fields(self):
        session_data = ChatSessionData(
            tab_id="tab-runtime",
            title="Runtime Session",
            conversation_id="conv-runtime",
            is_ephemeral=False,
            runtime_backend="server",
            discovery_owner="ccp_persona",
            discovery_entity_id="persona.remote.helper",
            assistant_kind="persona",
            assistant_id="persona.remote.helper",
        )

        restored = ChatSessionData.from_dict(session_data.to_dict())

        assert restored.runtime_backend == "server"
        assert restored.discovery_owner == "ccp_persona"
        assert restored.discovery_entity_id == "persona.remote.helper"
        assert restored.assistant_id == "persona.remote.helper"

    def test_chat_session_data_round_trip_preserves_handoff_payload(self):
        payload = ChatHandoffPayload(
            source="notes",
            item_type="note",
            title="Planning note",
            body="Plan content",
            source_id="note-1",
        )
        session = ChatSessionData(tab_id="tab-1", handoff_payload=payload)

        restored = ChatSessionData.from_dict(session.to_dict())

        assert restored.handoff_payload is not None
        assert restored.handoff_payload.title == "Planning note"


class TestMessageDataSerialization:
    def test_message_data_round_trip_preserves_tree_and_variant_fields(self):
        message = MessageData(
            message_id="msg-1",
            role="assistant",
            content="Response variant",
            timestamp=datetime(2026, 4, 19, 10, 30, 0),
            parent_message_id="msg-root",
            variant_of="msg-base",
            variant_number=2,
            is_selected_variant=True,
            total_variants=3,
        )

        restored = MessageData.from_dict(message.to_dict())

        assert restored.parent_message_id == "msg-root"
        assert restored.variant_of == "msg-base"
        assert restored.variant_number == 2
        assert restored.is_selected_variant is True
        assert restored.total_variants == 3

    def test_message_data_from_dict_preserves_missing_legacy_timestamp(self):
        restored = MessageData.from_dict(
            {
                "message_id": "msg-legacy",
                "role": "assistant",
                "content": "Legacy message",
            }
        )

        assert restored.timestamp is None


class TestTabStateSerialization:
    def test_tab_state_round_trip_preserves_assistant_scope_and_message_topology(self):
        tab_state = TabState(
            tab_id="tab-1",
            title="Prompt Session",
            conversation_id="conv-1",
            runtime_backend="server",
            discovery_owner="ccp_persona",
            discovery_entity_id="persona.remote.helper",
            assistant_kind="persona",
            assistant_id="assistant-1",
            persona_memory_mode="workspace",
            scope_type="workspace",
            workspace_id="workspace-1",
            messages=[
                MessageData(
                    message_id="msg-1",
                    role="assistant",
                    content="Hello",
                    timestamp=datetime(2026, 4, 19, 11, 0, 0),
                    parent_message_id="msg-root",
                    variant_of="msg-base",
                    variant_number=1,
                    is_selected_variant=True,
                    total_variants=2,
                )
            ],
        )

        restored = TabState.from_dict(tab_state.to_dict())

        assert restored.runtime_backend == "server"
        assert restored.discovery_owner == "ccp_persona"
        assert restored.discovery_entity_id == "persona.remote.helper"
        assert restored.assistant_kind == "persona"
        assert restored.assistant_id == "assistant-1"
        assert restored.persona_memory_mode == "workspace"
        assert restored.scope_type == "workspace"
        assert restored.workspace_id == "workspace-1"
        assert restored.messages[0].parent_message_id == "msg-root"
        assert restored.messages[0].variant_of == "msg-base"
        assert restored.messages[0].variant_number == 1
        assert restored.messages[0].is_selected_variant is True
        assert restored.messages[0].total_variants == 2

    def test_tab_state_round_trip_preserves_handoff_payload(self):
        payload = ChatHandoffPayload(
            source="media",
            item_type="media",
            title="Video",
            body="Transcript",
            source_id="media-1",
        )
        tab_state = TabState(tab_id="tab-1", title="Media: Video", handoff_payload=payload)

        restored = TabState.from_dict(tab_state.to_dict())

        assert restored.handoff_payload is not None
        assert restored.handoff_payload.source == "media"


class TestChatScreenStateSerialization:
    def test_chat_screen_state_round_trip_preserves_expanded_tab_and_message_fields(self):
        state = ChatScreenState(
            tabs=[
                TabState(
                    tab_id="tab-1",
                    title="Scoped Prompt",
                    runtime_backend="server",
                    discovery_owner="ccp_persona",
                    discovery_entity_id="persona.remote.helper",
                    assistant_kind="persona",
                    assistant_id="assistant-1",
                    persona_memory_mode="session",
                    scope_type="workspace",
                    workspace_id="workspace-5",
                    messages=[
                        MessageData(
                            message_id="msg-1",
                            role="assistant",
                            content="Variant response",
                            timestamp=datetime(2026, 4, 19, 12, 0, 0),
                            parent_message_id="msg-parent",
                            variant_of="msg-sibling-root",
                            variant_number=3,
                            is_selected_variant=False,
                            total_variants=4,
                        )
                    ],
                )
            ],
            active_tab_id="tab-1",
            tab_order=["tab-1"],
        )

        restored = ChatScreenState.from_dict(state.to_dict())
        restored_tab = restored.tabs[0]
        restored_message = restored_tab.messages[0]

        assert restored_tab.runtime_backend == "server"
        assert restored_tab.discovery_owner == "ccp_persona"
        assert restored_tab.discovery_entity_id == "persona.remote.helper"
        assert restored_tab.assistant_kind == "persona"
        assert restored_tab.assistant_id == "assistant-1"
        assert restored_tab.persona_memory_mode == "session"
        assert restored_tab.scope_type == "workspace"
        assert restored_tab.workspace_id == "workspace-5"
        assert restored_message.parent_message_id == "msg-parent"
        assert restored_message.variant_of == "msg-sibling-root"
        assert restored_message.variant_number == 3
        assert restored_message.is_selected_variant is False
        assert restored_message.total_variants == 4

    def test_tab_state_from_dict_defaults_missing_scope_to_global(self):
        restored = TabState.from_dict(
            {
                "tab_id": "tab-1",
                "title": "Generic Session",
                "assistant_kind": "persona",
                "assistant_id": "assistant-1",
                "workspace_id": "workspace-should-drop",
            }
        )

        assert restored.scope_type == "global"
        assert restored.workspace_id is None


class TestTabStateManager:
    @pytest.mark.asyncio
    async def test_create_tab_state_uses_explicit_assistant_and_scope_fields(self):
        manager = TabStateManager()

        state = await manager.create_tab_state(
            "tab-1",
            runtime_backend="server",
            discovery_owner="ccp_persona",
            discovery_entity_id="persona.remote.helper",
            assistant_kind="persona",
            assistant_id="assistant-1",
            persona_memory_mode="workspace",
            scope_type="workspace",
            workspace_id="workspace-1",
            unknown_flag="kept-in-metadata",
        )

        assert state.runtime_backend == "server"
        assert state.discovery_owner == "ccp_persona"
        assert state.discovery_entity_id == "persona.remote.helper"
        assert state.assistant_kind == "persona"
        assert state.assistant_id == "assistant-1"
        assert state.persona_memory_mode == "workspace"
        assert state.scope_type == "workspace"
        assert state.workspace_id == "workspace-1"
        assert "unknown_flag" not in state.__dict__
        assert state.metadata["unknown_flag"] == "kept-in-metadata"


class TestChatScreenRestore:
    @pytest.mark.asyncio
    async def test_restore_tab_sessions_does_not_overwrite_authoritative_runtime_source(self):
        mock_app = Mock()
        mock_app.get_current_screen_state = Mock(return_value={})
        mock_app.notify = Mock()
        mock_app.runtime_policy = Mock(state=RuntimeSourceState(active_source="local"))
        mock_app.current_runtime_backend = "local"
        mock_app.runtime_backend = "local"

        screen = ChatScreen(mock_app)
        screen.chat_state = ChatScreenState(
            tabs=[
                TabState(
                    tab_id="saved-tab-1",
                    title="Remote Session",
                    conversation_id="conv-restore",
                    runtime_backend="server",
                    is_ephemeral=False,
                ),
            ],
            active_tab_id="saved-tab-1",
            tab_order=["saved-tab-1"],
        )

        default_session = Mock()
        default_session.session_data = ChatSessionData(tab_id="default", title="Default")

        tab_container = Mock()
        tab_container.sessions = {"default": default_session}
        tab_container.close_tab = AsyncMock()
        tab_container.create_new_tab = AsyncMock(return_value="saved-tab-1")
        tab_container.switch_to_tab = AsyncMock()

        await screen._restore_tab_sessions(tab_container)

        assert mock_app.runtime_policy.state.active_source == "local"

    @pytest.mark.asyncio
    async def test_restore_tab_sessions_preserves_first_duplicate_and_remaps_active_tab(self):
        mock_app = Mock()
        mock_app.get_current_screen_state = Mock(return_value={})
        mock_app.notify = Mock()

        screen = ChatScreen(mock_app)
        screen.chat_state = ChatScreenState(
            tabs=[
                TabState(
                    tab_id="saved-tab-1",
                    title="First Runtime Session",
                    conversation_id="conv-restore",
                    runtime_backend="server",
                    discovery_owner="general_chat",
                    discovery_entity_id="assistant.remote.restore",
                    is_ephemeral=False,
                ),
                TabState(
                    tab_id="saved-tab-2",
                    title="Second Runtime Session",
                    conversation_id="conv-restore",
                    runtime_backend="server",
                    discovery_owner="general_chat",
                    discovery_entity_id="assistant.remote.restore",
                    is_ephemeral=False,
                ),
            ],
            active_tab_id="saved-tab-2",
            tab_order=["saved-tab-1", "saved-tab-2"],
        )

        restored_session = Mock()
        restored_session.session_data = ChatSessionData(tab_id="live-tab-1", title="placeholder")

        async def fake_create_new_tab(title=None, session_data=None):
            if "live-tab-1" not in tab_container.sessions:
                tab_container.sessions["live-tab-1"] = restored_session
                if session_data is not None:
                    restored_session.session_data = session_data
            return "live-tab-1"

        tab_container = Mock()
        tab_container.sessions = {}
        tab_container.close_tab = AsyncMock()
        tab_container.create_new_tab = AsyncMock(side_effect=fake_create_new_tab)

        await screen._restore_tab_sessions(tab_container)

        assert restored_session.session_data.title == "First Runtime Session"
        assert restored_session.session_data.conversation_id == "conv-restore"
        assert restored_session.session_data.runtime_backend == "server"
        assert screen.chat_state.active_tab_id == "live-tab-1"

    @pytest.mark.asyncio
    async def test_restore_messages_shows_chat_log_when_messages_are_restored(self):
        mock_app = Mock()
        mock_app.get_current_screen_state = Mock(return_value={})
        mock_app.notify = Mock()

        screen = ChatScreen(mock_app)
        screen.chat_state = ChatScreenState(
            tabs=[
                TabState(
                    tab_id="default",
                    title="Chat",
                    is_active=True,
                    messages=[
                        MessageData(
                            message_id="msg-1",
                            role="user",
                            content="hello",
                            timestamp=datetime(2026, 4, 21, 12, 0, 0),
                        )
                    ],
                )
            ],
            active_tab_id="default",
            tab_order=["default"],
        )

        chat_log = Mock()
        chat_log.mount = AsyncMock()
        chat_log.scroll_end = Mock()
        mock_app.query_one = Mock(return_value=chat_log)

        screen.chat_window = Mock()
        screen.chat_window.hide_empty_state = Mock()
        screen.chat_window.query = Mock(return_value=[])

        await screen._restore_messages()

        chat_log.mount.assert_awaited()
        chat_log.scroll_end.assert_called_once_with(animate=False)
        screen.chat_window.hide_empty_state.assert_called_once()
