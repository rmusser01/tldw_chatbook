"""P1g: the shared union core + summary projection."""

import json
import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    _resolve_active_dictionaries,
    summarize_active_dictionaries,
    collect_active_chatdict_entries,
    conversation_dictionary_ids,
)


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "resolve.db", "test-client")


def _attach_conv_dict(db, service, conv_id, name, enabled=True):
    dict_id = service.create_dictionary({"name": name, "entries": [{"pattern": name, "replacement": name.lower()}]})["id"]
    if not enabled:
        rec = service.get_dictionary(dict_id)
        service.update_dictionary(dict_id, {"is_active": False}, expected_version=rec["version"])
    conv = db.get_conversation_by_id(conv_id)
    meta = json.loads(conv.get("metadata") or "{}")
    meta["active_dictionaries"] = meta.get("active_dictionaries", []) + [dict_id]
    db.update_conversation(conv_id, {"metadata": json.dumps(meta)}, expected_version=conv["version"])
    return dict_id


def _embed_char_dict(db, char_id, name, enabled=True, entries=None):
    rec = db.get_character_card_by_id(char_id)
    ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    ext.setdefault("chat_dictionaries", []).append(
        {"name": name, "enabled": enabled, "entries": entries or [{"key": name, "content": name.lower()}]}
    )
    db.update_character_card(char_id, {"extensions": ext}, expected_version=rec["version"])


def test_summary_applied_set_equals_collect(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "ConvDict")
    char_id = db.add_character_card({"name": "Noir"})
    _embed_char_dict(db, char_id, "CharDict")
    char_data = db.get_character_card_by_id(char_id)

    summary = summarize_active_dictionaries(db, conv_id, char_data)["dictionaries"]
    applied_names_summary = {d["name"] for d in summary if d["enabled"] and not d["shadowed"]}
    # The applied set from the summary must equal the dicts collect actually loads.
    collect_names = {"ConvDict", "CharDict"}
    assert applied_names_summary == collect_names
    assert len(collect_active_chatdict_entries(db, conv_id, char_data)) == 2  # one entry each


def test_shadowed_only_by_enabled_conversation_dict(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "Shared", enabled=True)
    char_id = db.add_character_card({"name": "Noir"})
    _embed_char_dict(db, char_id, "Shared")  # same name as an ENABLED conversation dict
    char_data = db.get_character_card_by_id(char_id)

    rows = _resolve_active_dictionaries(db, conv_id, char_data)
    char_shared = [r for r in rows if r["source"] == "character" and r["name"] == "Shared"][0]
    assert char_shared["shadowed"] is True
    # Only the conversation "Shared" applies.
    assert collect_active_chatdict_entries(db, conv_id, char_data)[0].content == "shared"


def test_not_shadowed_by_disabled_conversation_dict(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "Shared", enabled=False)  # DISABLED
    char_id = db.add_character_card({"name": "Noir"})
    _embed_char_dict(db, char_id, "Shared", entries=[{"key": "s", "content": "CHAR"}])
    char_data = db.get_character_card_by_id(char_id)

    rows = _resolve_active_dictionaries(db, conv_id, char_data)
    char_shared = [r for r in rows if r["source"] == "character" and r["name"] == "Shared"][0]
    assert char_shared["shadowed"] is False  # disabled conv dict does NOT shadow
    # The character "Shared" applies (the disabled conversation one does not).
    assert collect_active_chatdict_entries(db, conv_id, char_data)[0].content == "CHAR"


def test_summary_entry_count_and_source_and_never_raises(db):
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    ext["chat_dictionaries"] = "not-a-list"  # hostile
    db.update_character_card(char_id, {"extensions": ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    # never raises on hostile embedded content
    assert summarize_active_dictionaries(db, None, char_data) == {"dictionaries": [], "source": "local"}


# --- P1g Task 5: conversation_dictionary_ids ---------------------------------

def test_conversation_dictionary_ids_reads_active_dictionaries_in_order(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    first = _attach_conv_dict(db, service, conv_id, "First")
    second = _attach_conv_dict(db, service, conv_id, "Second")

    assert conversation_dictionary_ids(db, conv_id) == [first, second]


def test_conversation_dictionary_ids_never_raises_on_missing_or_none(db):
    assert conversation_dictionary_ids(db, None) == []
    assert conversation_dictionary_ids(None, "some-id") == []
    assert conversation_dictionary_ids(db, "does-not-exist") == []


def test_conversation_dictionary_ids_tolerates_non_dict_metadata(db):
    """Regression companion to the P1e ``attach_survives_non_dict_metadata_json``
    finding: a conversation's ``metadata`` can be valid JSON but not a JSON
    object (e.g. a bare scalar). Must degrade to ``[]`` instead of raising."""
    conv_id = db.add_conversation({"title": "chat"})
    rec = db.get_conversation_by_id(conv_id)
    db.update_conversation(conv_id, {"metadata": "5"}, expected_version=rec["version"])

    assert conversation_dictionary_ids(db, conv_id) == []


def test_conversation_dictionary_ids_coerces_and_dedups(db):
    conv_id = db.add_conversation({"title": "chat"})
    rec = db.get_conversation_by_id(conv_id)
    db.update_conversation(
        conv_id,
        {"metadata": json.dumps({"active_dictionaries": [3, "3", "not-an-id", 5, 3]})},
        expected_version=rec["version"],
    )

    assert conversation_dictionary_ids(db, conv_id) == [3, 5]
