"""P1e: the dead dictionary-attach code paths are gone and nothing imports them."""

import importlib

import pytest


def test_chat_events_dictionaries_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_chatbook.Event_Handlers.Chat_Events.chat_events_dictionaries")


def test_dead_junction_functions_removed():
    import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl
    assert not hasattr(cdl, "associate_dictionary_with_conversation")
    assert not hasattr(cdl, "get_conversation_dictionaries")


def test_app_and_chat_events_still_import():
    # The wiring removal didn't break the modules that referenced the dead handlers.
    importlib.import_module("tldw_chatbook.Event_Handlers.Chat_Events.chat_events")
    importlib.import_module("tldw_chatbook.Event_Handlers.conv_char_events")
    importlib.import_module("tldw_chatbook.Event_Handlers.event_dispatcher")
    importlib.import_module("tldw_chatbook.app")
