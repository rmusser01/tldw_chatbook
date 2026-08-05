"""Load-bearing integration test: the native Console send path applies the
active CONVERSATION world-info to the model-bound payload while the
persisted transcript keeps the raw text (Roleplay P2g-1 Task 3).

Exercises the real ``ChatScreen`` -> ``_ensure_console_chat_controller``
wiring (``world_info_applier=self._console_world_info_applier``) end to end:
a real ``CharactersRAGDB`` seeded with a conversation-attached world book (the
``WorldBookManager`` attach seam), a native session pinned to that
conversation, and a capturing double standing in for the provider transport
so the actual outbound payload can be inspected.

Mirrors ``Tests/UI/test_console_dictionary_send_integration.py``'s harness,
swapping the chat-dictionary seam for the world-book seam.
"""

import pytest

from Tests.UI.test_console_dictionary_send_integration import (
    _CapturingGateway,
    _final_user_content,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "console_world_info_send.db", "test-client")
    yield db
    db.close_connection()


def _active_native_session(console):
    store = console._ensure_console_chat_store()
    return next(s for s in store.sessions() if s.id == store.active_session_id)


@pytest.mark.asyncio
async def test_native_send_applies_conversation_world_info_provider_branch(wb_db):
    app = _build_test_app()
    app.chachanotes_db = wb_db

    conv_id = wb_db.add_conversation({"title": "World send"})
    wb = WorldBookManager(wb_db)
    book_id = wb.create_world_book("Lore")
    wb.create_world_book_entry(
        book_id, keys=["dragon"], content="Dragons breathe fire."
    )
    wb.associate_world_book_with_conversation(conv_id, book_id)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False  # force the provider branch

        result = await controller.submit_draft("a dragon appears")
        assert result.accepted

        # Model received the world-info-INJECTED text...
        final_content = _final_user_content(gateway.captured)
        assert "Dragons breathe fire." in final_content
        assert "a dragon appears" in final_content

        # ...while the persisted transcript keeps the RAW text.
        store = screen._ensure_console_chat_store()
        stored = [
            m
            for m in store.messages_for_session(_active_native_session(screen).id)
            if m.role is ConsoleMessageRole.USER
        ]
        assert stored[-1].content == "a dragon appears"


@pytest.mark.asyncio
async def test_native_send_world_info_disabled_by_config_not_injected(wb_db, monkeypatch):
    app = _build_test_app()
    app.chachanotes_db = wb_db

    conv_id = wb_db.add_conversation({"title": "World send disabled"})
    wb = WorldBookManager(wb_db)
    book_id = wb.create_world_book("Lore")
    wb.create_world_book_entry(
        book_id, keys=["dragon"], content="Dragons breathe fire."
    )
    wb.associate_world_book_with_conversation(conv_id, book_id)

    from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

    def _fake_get_cli_setting(section, key, default=None):
        if section == "character_chat" and key == "enable_world_info":
            return False
        return default

    monkeypatch.setattr(
        chat_screen_module, "get_cli_setting", _fake_get_cli_setting
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False  # force the provider branch

        result = await controller.submit_draft("a dragon appears")
        assert result.accepted

        final_content = _final_user_content(gateway.captured)
        assert final_content == "a dragon appears"
        assert "Dragons breathe fire." not in final_content


def test_console_world_info_applier_honors_enable_world_info_setting(monkeypatch):
    """Focused unit test of the gate itself (belt-and-braces alongside the
    end-to-end disabled-config test above): the bound applier must return the
    text unchanged, without ever reaching ``apply_world_info_to_message``,
    when ``[character_chat] enable_world_info`` is falsy."""
    from tldw_chatbook.Character_Chat import world_info_resolver
    from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

    def _fake_get_cli_setting(section, key, default=None):
        if section == "character_chat" and key == "enable_world_info":
            return False
        return default

    monkeypatch.setattr(
        chat_screen_module, "get_cli_setting", _fake_get_cli_setting
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "apply_world_info_to_message must not be called when "
            "enable_world_info is disabled"
        )

    monkeypatch.setattr(
        world_info_resolver, "apply_world_info_to_message", _fail_if_called
    )

    class _FakeApp:
        chachanotes_db = object()  # non-None so the earlier guards pass

    class _FakeScreen:
        app_instance = _FakeApp()

    # Bind the real method to a lightweight stand-in with just app_instance.
    applier = chat_screen_module.ChatScreen._console_world_info_applier.__get__(
        _FakeScreen()
    )
    result = applier("conv-1", "a dragon appears", [])
    assert result == "a dragon appears"
