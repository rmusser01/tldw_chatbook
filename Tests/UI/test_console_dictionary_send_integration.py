"""Load-bearing integration test: the native Console send path applies the
active CONVERSATION chat dictionaries to the model-bound payload while the
persisted transcript keeps the raw text (Roleplay P1h Task 3).

Exercises the real ``ChatScreen`` -> ``_ensure_console_chat_controller`` wiring
(``chat_dictionary_applier=self._console_chat_dictionary_applier``) end to
end: a real ``CharactersRAGDB`` + ``ChatDictionaryScopeService`` seeded with a
conversation-attached dictionary (the P1e attach seam), a native session
pinned to that conversation, and a capturing double standing in for the
provider/agent transport so the actual outbound payload can be inspected.
"""
import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def dictionary_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "console_dict_send.db", "test-client")
    yield db
    db.close_connection()


def _active_native_session(console):
    store = console._ensure_console_chat_store()
    return next(s for s in store.sessions() if s.id == store.active_session_id)


class _CapturingGateway:
    """Records the provider_messages the send would transmit; yields one chunk."""

    def __init__(self):
        self.captured = None

    async def resolve_for_send(self, _selection):
        class _R:
            ready = True
            visible_copy = ""
        return _R()

    async def stream_chat(self, _resolution, provider_messages):
        self.captured = [dict(m) for m in provider_messages]
        yield "ok"


def _final_user_content(messages):
    for message in reversed(messages):
        if message.get("role") == ConsoleMessageRole.USER.value:
            return message.get("content")
    return None


@pytest.mark.asyncio
async def test_native_send_applies_conversation_dictionary_provider_branch(dictionary_db):
    app = _build_test_app()
    app.chachanotes_db = dictionary_db
    app.chat_dictionary_scope_service = ChatDictionaryScopeService(
        local_service=LocalChatDictionaryService(dictionary_db), server_service=None
    )

    conv_id = dictionary_db.add_conversation({"title": "Send flow"})
    dict_id = cdl.save_chat_dictionary(
        dictionary_db, "Slang", entries=[cdl.ChatDictionary(key="Warden", content="grim jailer")]
    )
    LocalChatDictionaryService(dictionary_db).attach_to_conversation(dict_id, conv_id)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False  # force the provider branch

        result = await controller.submit_draft("The Warden nods.")
        assert result.accepted

        # Model received the SUBSTITUTED text...
        assert _final_user_content(gateway.captured) == "The grim jailer nods."
        # ...while the persisted transcript keeps the RAW text.
        store = screen._ensure_console_chat_store()
        stored = [m for m in store.messages_for_session(_active_native_session(screen).id)
                  if m.role is ConsoleMessageRole.USER]
        assert stored[-1].content == "The Warden nods."


@pytest.mark.asyncio
async def test_native_send_applies_conversation_dictionary_agent_branch(dictionary_db):
    app = _build_test_app()
    app.chachanotes_db = dictionary_db
    app.chat_dictionary_scope_service = ChatDictionaryScopeService(
        local_service=LocalChatDictionaryService(dictionary_db), server_service=None
    )

    conv_id = dictionary_db.add_conversation({"title": "Agent send"})
    dict_id = cdl.save_chat_dictionary(
        dictionary_db, "Slang", entries=[cdl.ChatDictionary(key="Warden", content="grim jailer")]
    )
    LocalChatDictionaryService(dictionary_db).attach_to_conversation(dict_id, conv_id)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway

        captured = {}

        # Mirrors the real `ConsoleAgentBridge.run_reply` contract (see
        # `Tests/Chat/test_console_agent_swap.py`'s canonical double): the
        # brief's originally sketched fake returned a `ConsoleSubmitResult`
        # directly, but `_run_agent_reply` (console_chat_controller.py)
        # awaits `asyncio.to_thread(self._agent_bridge.run_reply, ...)` and
        # feeds the RETURN VALUE into `_finalize_agent_reply`, which reads
        # `outcome.status` against `RunOutcome`'s `RUN_DONE`/`RUN_CANCELLED`
        # sentinels -- not a `ConsoleSubmitResult`. This fake appends the
        # streamed content to the store itself (as the real bridge does
        # internally) and returns a `RunOutcome(status=RUN_DONE, ...)` so
        # `_finalize_agent_reply`'s success path (`store.mark_message_complete`)
        # runs unmodified against a real placeholder message.
        def _fake_run_reply(*, conversation_id, session_id, resolution, assistant_message_id,
                             model, session_system_prompt, agent_messages, should_cancel,
                             supersede_previous=False):
            captured["agent_messages"] = [dict(m) for m in agent_messages]
            from tldw_chatbook.Agents.agent_models import RunOutcome, RUN_DONE
            store = screen._ensure_console_chat_store()
            store.append_stream_chunk(assistant_message_id, "ok")
            return RunOutcome(status=RUN_DONE, steps=[])

        class _Bridge:
            run_reply = staticmethod(_fake_run_reply)

        controller._agent_bridge = _Bridge()
        controller._agent_runtime_enabled = True

        result = await controller.submit_draft("The Warden nods.")
        assert result.accepted
        assert _final_user_content(captured["agent_messages"]) == "The grim jailer nods."
