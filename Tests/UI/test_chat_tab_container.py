from unittest.mock import AsyncMock, Mock, patch

import pytest

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer


def _make_session(session_data: ChatSessionData) -> Mock:
    session = Mock()
    session.session_data = session_data
    session.styles = Mock()
    session.resume = AsyncMock()
    session.suspend = AsyncMock()
    session.cleanup = AsyncMock()
    session.remove = AsyncMock()
    return session


class TestChatTabContainerShellSync:
    def test_container_reads_chat_defaults_max_tabs(self, monkeypatch):
        app = Mock()

        def fake_get_cli_setting(section, key, default=None):
            if section == "chat_defaults" and key == "max_tabs":
                return 4
            return default

        monkeypatch.setattr(
            "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container.get_cli_setting",
            fake_get_cli_setting,
        )

        container = ChatTabContainer(app)

        assert container.max_tabs == 4

    @pytest.mark.asyncio
    async def test_create_new_tab_reuse_publishes_active_session(self):
        app = Mock()
        app.notify = Mock()
        app.call_later = Mock()

        existing_session_data = ChatSessionData(
            tab_id="aaaaaaaa",
            title="Existing Conversation",
            conversation_id="conv-1",
            runtime_backend="server",
        )
        existing_session = _make_session(existing_session_data)

        container = ChatTabContainer(app)
        container.sessions = {"aaaaaaaa": existing_session}
        container.active_session_id = "aaaaaaaa"
        container.post_message = Mock()

        reused_id = await container.create_new_tab(
            session_data=ChatSessionData(
                tab_id="ignored",
                title="Reuse Request",
                conversation_id="conv-1",
                runtime_backend="server",
            )
        )

        assert reused_id == "aaaaaaaa"
        message = container.post_message.call_args.args[0]
        assert isinstance(message, ChatTabContainer.ActiveSessionChanged)
        assert message.session_data is existing_session_data

    @pytest.mark.asyncio
    async def test_switch_to_tab_async_publishes_active_session(self):
        app = Mock()
        app.notify = Mock()
        app.call_later = Mock()

        first_session = _make_session(
            ChatSessionData(tab_id="aaaaaaaa", title="First Session")
        )
        second_session_data = ChatSessionData(
            tab_id="bbbbbbbb",
            title="Persona Session",
            runtime_backend="server",
            assistant_kind="persona",
            assistant_id="study.coach",
            scope_type="workspace",
            workspace_id="workspace-42",
        )
        second_session = _make_session(second_session_data)

        container = ChatTabContainer(app)
        container.sessions = {
            "aaaaaaaa": first_session,
            "bbbbbbbb": second_session,
        }
        container.active_session_id = "aaaaaaaa"
        container.tab_bar = Mock()
        container.post_message = Mock()

        await container.switch_to_tab_async("bbbbbbbb")

        message = container.post_message.call_args.args[0]
        assert isinstance(message, ChatTabContainer.ActiveSessionChanged)
        assert message.session_data is second_session_data
        assert container.active_session_id == "bbbbbbbb"

    @pytest.mark.asyncio
    async def test_close_active_tab_publishes_next_active_session(self):
        app = Mock()
        app.notify = Mock()
        app.call_later = Mock()

        first_session = _make_session(
            ChatSessionData(tab_id="aaaaaaaa", title="First Session")
        )
        second_session_data = ChatSessionData(
            tab_id="bbbbbbbb",
            title="Fallback Session",
            runtime_backend="server",
            assistant_kind="character",
            character_name="Navigator",
        )
        second_session = _make_session(second_session_data)

        container = ChatTabContainer(app)
        container.sessions = {
            "aaaaaaaa": first_session,
            "bbbbbbbb": second_session,
        }
        container.active_session_id = "aaaaaaaa"
        container.tab_bar = Mock()
        container.post_message = Mock()

        await container._force_close_tab("aaaaaaaa")

        message = container.post_message.call_args.args[0]
        assert isinstance(message, ChatTabContainer.ActiveSessionChanged)
        assert message.session_data is second_session_data
        assert container.active_session_id == "bbbbbbbb"

    @pytest.mark.asyncio
    async def test_close_last_tab_clears_active_session(self):
        app = Mock()
        app.notify = Mock()
        app.call_later = Mock()

        only_session = _make_session(
            ChatSessionData(tab_id="aaaaaaaa", title="Only Session")
        )
        placeholder = Mock()
        placeholder.styles = Mock()

        container = ChatTabContainer(app)
        container.sessions = {"aaaaaaaa": only_session}
        container.active_session_id = "aaaaaaaa"
        container.tab_bar = Mock()
        container.query_one = Mock(return_value=placeholder)
        container.post_message = Mock()

        await container._force_close_tab("aaaaaaaa")

        message = container.post_message.call_args.args[0]
        assert isinstance(message, ChatTabContainer.ActiveSessionChanged)
        assert message.session_data is None
        assert container.active_session_id is None

    @pytest.mark.asyncio
    async def test_handoff_session_with_no_conversation_id_does_not_reuse_existing_tab(self):
        app = Mock()
        app.notify = Mock()
        app.call_later = Mock()
        existing = _make_session(
            ChatSessionData(
                tab_id="aaaaaaaa",
                title="Existing",
                conversation_id="conv-1",
                runtime_backend="server",
            )
        )
        container = ChatTabContainer(app)
        container.sessions = {"aaaaaaaa": existing}
        container.max_tabs = 10
        mount_target = Mock()
        mount_target.mount = AsyncMock()
        container.query_one = Mock(return_value=mount_target)
        container.post_message = Mock()

        session_data = ChatSessionData(
            tab_id="handoff",
            title="Note: Plan",
            conversation_id=None,
            runtime_backend="server",
            handoff_payload=ChatHandoffPayload(
                source="notes",
                item_type="note",
                title="Plan",
                body="Body",
            ),
        )

        with patch(
            "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container.ChatSession",
            return_value=_make_session(session_data),
        ):
            tab_id = await container.create_new_tab(session_data=session_data)

        assert tab_id != "aaaaaaaa"
