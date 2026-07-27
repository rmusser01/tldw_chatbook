"""P1e: the dead dictionary-attach code paths are gone and nothing imports them."""

import importlib

import pytest


def test_chat_events_dictionaries_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_dictionaries"
        )


def test_event_dispatcher_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_chatbook.Event_Handlers.event_dispatcher")


def test_dead_junction_functions_removed():
    import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl

    assert not hasattr(cdl, "associate_dictionary_with_conversation")
    assert not hasattr(cdl, "get_conversation_dictionaries")


def test_app_imports_without_retired_conv_char_events():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_chatbook.Event_Handlers.conv_char_events")
    importlib.import_module("tldw_chatbook.app")


def test_chat_events_module_removed():
    # task-577 PR2 T3: chat_events.py was deleted outright (its keep-set was
    # empty -- every external caller was dead or died in Phase 1).
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_chatbook.Event_Handlers.Chat_Events.chat_events")
