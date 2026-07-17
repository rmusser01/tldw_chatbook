"""P1g Task 5: Console attach/detach handlers + picker wiring for
CONVERSATION-scoped chat dictionaries.

Real-seam round-trip: combines the native-session Console harness
(``ConsoleHarness``/``_build_test_app``, as in
``Tests/UI/test_console_dictionaries_screen.py``) with a REAL DB +
REAL ``ChatDictionaryScopeService(local_service=LocalChatDictionaryService(db))``
(as in ``Tests/Character_Chat/test_local_chat_dictionary_service.py``). A
fake scope service cannot prove attach genuinely writes
``conversations.metadata.active_dictionaries`` -- only a real DB round-trip
can.

The active conversation is sourced from ``ChatScreen`` accessor
``_current_console_rail_conversation_id()`` (the active native Console
session's ``persisted_conversation_id``) -- NEVER from the app-level
``current_chat_conversation_id`` reactive, which the native Console never
writes (see Task 4's Critical / the documented Console<->Library
split-brain).
"""

import json

import pytest
from textual.widgets import Button

from Tests.UI.test_console_internals_decomposition import _open_console_inspector
from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_console_dictionaries import (
    console_attached_dictionaries,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker


@pytest.fixture
def dictionary_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "console_dictionaries_attach.db", "test-client")
    yield db
    db.close_connection()


def _active_native_session(console: ChatScreen):
    store = console._ensure_console_chat_store()
    return next(s for s in store.sessions() if s.id == store.active_session_id)


def _active_dictionaries(db, conversation_id) -> list:
    record = db.get_conversation_by_id(conversation_id)
    meta = json.loads((record or {}).get("metadata") or "{}")
    return meta.get("active_dictionaries", [])


async def _press_and_await_worker(pilot, screen, selector: str) -> None:
    """Press a Console inspector action button and wait for its worker.

    Uses ``Button.press()`` rather than ``pilot.click()``: the dictionary
    actions render at the tail of a long ``VerticalScroll`` inspector body
    (past several other row groups), where real screen-coordinate hit
    testing is unreliable across a recompose. ``.press()`` is the pattern
    already used for deep Console buttons elsewhere (see
    ``test_console_native_chat_flow.py``'s ``#console-send-message``).
    """
    screen.query_one(selector, Button).press()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


# --- Real-DB round-trip (load-bearing) --------------------------------------

@pytest.mark.asyncio
async def test_console_attach_then_detach_round_trips_through_real_db(dictionary_db, monkeypatch):
    app = _build_test_app()
    local_service = LocalChatDictionaryService(dictionary_db)
    scope_service = ChatDictionaryScopeService(local_service=local_service, server_service=None)
    app.chachanotes_db = dictionary_db
    app.chat_dictionary_scope_service = scope_service

    conv_id = dictionary_db.add_conversation({"title": "Attach flow"})
    dict_id = cdl.save_chat_dictionary(dictionary_db, "Slang")
    assert dict_id is not None

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        # Select a configured provider so the first-run setup overlay isn't
        # covering the workbench -- unrelated to this feature, but required
        # for the inspector's action buttons to actually be reachable.
        _select_llamacpp_console(screen)
        await pilot.pause()
        await _open_console_inspector(screen, pilot)
        _active_native_session(screen).persisted_conversation_id = conv_id

        async def _fake_push_screen_wait(picker):
            return dict_id if isinstance(picker, DictionaryPicker) else None

        monkeypatch.setattr(screen.app_instance, "push_screen_wait", _fake_push_screen_wait, raising=False)

        await screen.refresh_active_dictionaries_summary()
        await pilot.pause()
        assert _active_dictionaries(dictionary_db, conv_id) == []

        # --- Attach ---
        await _press_and_await_worker(pilot, screen, "#console-inspector-dictionaries-attach")

        assert cdl.conversation_dictionary_ids(dictionary_db, conv_id) == [dict_id]
        assert _active_dictionaries(dictionary_db, conv_id) == [dict_id]

        summary = screen._active_dictionaries_summary
        assert summary is not None
        entries = summary.get("dictionaries") or []
        assert len(entries) == 1
        assert entries[0]["name"] == "Slang"
        assert entries[0]["source"] == "conversation"

        # --- Detach (same monkeypatched picker returns the same id) ---
        await _press_and_await_worker(pilot, screen, "#console-inspector-dictionaries-detach")

        assert cdl.conversation_dictionary_ids(dictionary_db, conv_id) == []
        assert _active_dictionaries(dictionary_db, conv_id) == []

        summary_after = screen._active_dictionaries_summary
        assert (summary_after or {}).get("dictionaries") == []


@pytest.mark.asyncio
async def test_console_attach_notifies_and_noops_without_a_conversation(dictionary_db, monkeypatch):
    """No active native-session conversation -> attach must not touch the DB
    or the (unrelated) app-level ``current_chat_conversation_id`` reactive.

    The attach action is *disabled* (and hidden) with no conversation
    (``_console_dictionary_inspector_actions``), so this drives the worker
    directly rather than clicking the rendered button -- exercising the same
    production guard the button-press handler defers to.
    """
    app = _build_test_app()
    local_service = LocalChatDictionaryService(dictionary_db)
    scope_service = ChatDictionaryScopeService(local_service=local_service, server_service=None)
    app.chachanotes_db = dictionary_db
    app.chat_dictionary_scope_service = scope_service

    calls: list = []
    monkeypatch.setattr(app, "notify", lambda *a, **k: calls.append((a, k)))

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        assert _active_native_session(screen).persisted_conversation_id is None

        actions = screen._console_dictionary_inspector_actions()
        attach_action = next(a for a in actions if a.widget_id == "console-inspector-dictionaries-attach")
        assert attach_action.enabled is False

        push_calls: list = []

        async def _fake_push_screen_wait(picker):
            push_calls.append(picker)
            return None

        monkeypatch.setattr(screen.app_instance, "push_screen_wait", _fake_push_screen_wait, raising=False)

        await screen._console_dictionary_attach_worker()
        await pilot.pause()

        assert push_calls == []  # never even opened the picker
        assert cdl.conversation_dictionary_ids(dictionary_db, "does-not-matter") == []
        assert any("conversation" in str(a).lower() for a, _k in calls)
        assert screen._console_dictionary_dialog_active is False


# --- Focused unit assertion: character-source dicts never leak in ----------

def test_console_attached_dictionaries_excludes_character_source_even_when_active(dictionary_db):
    """``console_attached_dictionaries`` must return ONLY conversation-source
    dictionaries -- even when a character dictionary is also active for the
    same DB. Character-scoped attachment is out of scope for the Console
    detach picker entirely; it must never appear there, whether or not one
    happens to be attached to some character in the same database."""
    service = LocalChatDictionaryService(dictionary_db)

    conv_id = dictionary_db.add_conversation({"title": "Mixed scope"})
    conv_dict_id = cdl.save_chat_dictionary(dictionary_db, "Conversation Lore")
    char_dict_id = cdl.save_chat_dictionary(dictionary_db, "Character Lore")
    assert conv_dict_id is not None and char_dict_id is not None

    service.attach_to_conversation(conv_dict_id, conv_id)
    char_id = dictionary_db.add_character_card({"name": "Noir"})
    service.attach_to_character(char_dict_id, char_id)

    rows = console_attached_dictionaries(dictionary_db, conv_id)

    assert [r["name"] for r in rows] == ["Conversation Lore"]
    assert all(r["dictionary_id"] != char_dict_id for r in rows)
