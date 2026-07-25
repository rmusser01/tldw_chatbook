from datetime import datetime
from unittest.mock import Mock

import pytest

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.tabs import TabContext
from tldw_chatbook.Chat.tabs.tab_state_manager import TabStateManager
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import (
    ChatScreenState,
    MessageData,
    TabState,
)


class EmptyChatLog:
    children = []

    def query(self, _selector):
        return []


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
        tab_state = TabState(
            tab_id="tab-1", title="Media: Video", handoff_payload=payload
        )

        restored = TabState.from_dict(tab_state.to_dict())

        assert restored.handoff_payload is not None
        assert restored.handoff_payload.source == "media"


class TestChatScreenStateSerialization:
    def test_chat_screen_state_round_trip_preserves_expanded_tab_and_message_fields(
        self,
    ):
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


class TestConsoleSessionSettingsPersonaLabelCompat:
    def test_restore_console_settings_accepts_pre_rename_persona_label_key(self):
        """task-442 T2 accept-old-write-new: a pre-rename serialized settings
        dict (``persona_label``) must deserialize into ``user_profile_label``
        -- old saved state predates the rename and must not silently lose the
        value (nor round-trip the old key back out)."""
        from dataclasses import asdict

        from tldw_chatbook.Chat.console_session_settings import (
            ConsoleSessionSettings,
        )

        settings = ConsoleSessionSettings(
            provider="openai", user_profile_label="Explorer"
        )
        old_blob = asdict(settings)
        old_blob["persona_label"] = old_blob.pop("user_profile_label")

        restored = ChatScreen._restore_console_settings(old_blob)

        assert restored is not None
        assert restored.user_profile_label == "Explorer"
        assert asdict(restored).get("persona_label") is None


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


def test_tab_context_uses_original_query_callable_after_monkey_patch(monkeypatch):
    selectors_seen = []
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key, default=None: (
            True if (section, key) == ("chat_defaults", "enable_tabs") else default
        ),
    )

    def original_query_one(selector, widget_type=None):
        selectors_seen.append((selector, widget_type))
        return f"resolved:{selector}"

    def recursive_query_one(_selector, _widget_type=None):
        raise AssertionError("patched app.query_one should not be called recursively")

    app = Mock()
    app.query_one = recursive_query_one
    context = TabContext(
        app,
        ChatSessionData(tab_id="tab-a"),
        query_one=original_query_one,
    )

    assert context.query_one("#chat-input") == "resolved:#chat-input-tab-a"
    assert selectors_seen == [("#chat-input-tab-a", None)]


def test_extract_messages_clears_messages_when_direct_chat_log_lookup_succeeds():
    app = Mock()
    app.query_one = Mock(return_value=EmptyChatLog())
    screen = ChatScreen(app)
    screen.chat_window = Mock()
    tab_state = TabState(
        tab_id="tab-1",
        title="Chat",
        messages=[
            MessageData(message_id="old", role="user", content="stale", timestamp=None)
        ],
    )

    screen._extract_and_save_messages(tab_state)

    assert tab_state.messages == []
    screen.chat_window.query.assert_not_called()


