from datetime import datetime

import pytest

from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.tabs.tab_state_manager import TabStateManager
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, MessageData, TabState


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
