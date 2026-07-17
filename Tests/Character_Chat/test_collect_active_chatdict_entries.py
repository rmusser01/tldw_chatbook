"""P1f: the runtime union (conversation + character dictionaries, dedup by name)."""

import json
import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import collect_active_chatdict_entries


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "collect.db", "test-client")


def _attach_conv_dict(db, service, conv_id, name):
    dict_id = service.create_dictionary({"name": name, "entries": [{"pattern": name, "replacement": name.lower()}]})["id"]
    conv = db.get_conversation_by_id(conv_id)
    meta = json.loads(conv.get("metadata") or "{}")
    meta["active_dictionaries"] = meta.get("active_dictionaries", []) + [dict_id]
    db.update_conversation(conv_id, {"metadata": json.dumps(meta)}, expected_version=conv["version"])
    return dict_id


def test_character_dictionary_fires_in_a_send(db):
    service = LocalChatDictionaryService(db)
    dict_id = service.create_dictionary({"name": "CharDict", "entries": [{"pattern": "hi", "replacement": "hello"}]})["id"]
    char_id = db.add_character_card({"name": "Noir"})
    service.attach_to_character(dict_id, char_id)
    char_data = db.get_character_card_by_id(char_id)

    entries = collect_active_chatdict_entries(db, None, char_data)

    assert len(entries) == 1
    assert entries[0].content == "hello"


def test_additive_union_conversation_wins_on_name_collision(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    # Conversation attaches "Shared" (live). Character embeds "Shared" (snapshot) + "Extra".
    _attach_conv_dict(db, service, conv_id, "Shared")
    char_id = db.add_character_card({"name": "Noir"})
    # Manually embed a same-named "Shared" snapshot to force the collision.
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = [
        {"name": "Shared", "enabled": True, "entries": [{"key": "s", "content": "CHAR"}]},
        {"name": "Extra", "enabled": True, "entries": [{"key": "e", "content": "x"}]},
    ]
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)

    entries = collect_active_chatdict_entries(db, conv_id, char_data)
    contents = sorted(e.content for e in entries)
    # "Shared" comes from the conversation (its content, "shared"), NOT the char snapshot ("CHAR");
    # "Extra" from the character. The char's "Shared" snapshot is dropped.
    assert "CHAR" not in contents
    assert "x" in contents  # Extra applied
    assert any(c == "shared" for c in contents)  # conversation Shared applied


def test_disabled_embedded_dictionary_is_skipped(db):
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = [{"name": "Off", "enabled": False, "entries": [{"key": "k", "content": "c"}]}]
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    assert collect_active_chatdict_entries(db, None, char_data) == []


def test_enabled_false_string_embedded_dictionary_is_skipped(db):
    """A malformed embedded block with enabled: "false" (string) must be
    skipped on the send path -- bool("false") is True and would wrongly
    apply it, violating "only enabled dicts contribute"."""
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = [{"name": "Off", "enabled": "false", "entries": [{"key": "k", "content": "c"}]}]
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    assert collect_active_chatdict_entries(db, None, char_data) == []


def test_malformed_embedded_content_never_raises(db):
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = "not-a-list"
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    assert collect_active_chatdict_entries(db, None, char_data) == []


def test_conversation_only_output_unchanged(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "OnlyConv")
    # No active character.
    entries = collect_active_chatdict_entries(db, conv_id, None)
    assert len(entries) == 1
    assert entries[0].content == "onlyconv"


def test_non_list_active_dictionaries_never_raises(db):
    conv_id = db.add_conversation({"title": "chat"})
    import json as _json
    conv = db.get_conversation_by_id(conv_id)
    db.update_conversation(conv_id, {"metadata": _json.dumps({"active_dictionaries": 5})},
                           expected_version=conv["version"])
    assert collect_active_chatdict_entries(db, conv_id, None) == []


def test_collect_disabled_conversation_dict_does_not_shadow_character(db):
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    import json as _json
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    did = service.create_dictionary({"name": "Dup", "entries": [{"pattern": "a", "replacement": "conv"}]})["id"]
    rec = service.get_dictionary(did)
    service.update_dictionary(did, {"is_active": False}, expected_version=rec["version"])  # disable it
    conv = db.get_conversation_by_id(conv_id)
    meta = _json.loads(conv.get("metadata") or "{}"); meta["active_dictionaries"] = [did]
    db.update_conversation(conv_id, {"metadata": _json.dumps(meta)}, expected_version=conv["version"])
    char_id = db.add_character_card({"name": "C"})
    r = db.get_character_card_by_id(char_id); ext = r["extensions"] if isinstance(r["extensions"], dict) else {}
    ext["chat_dictionaries"] = [{"name": "Dup", "enabled": True, "entries": [{"key": "a", "content": "char"}]}]
    db.update_character_card(char_id, {"extensions": ext}, expected_version=r["version"])
    entries = collect_active_chatdict_entries(db, conv_id, db.get_character_card_by_id(char_id))
    assert [e.content for e in entries] == ["char"]  # character applies; disabled conv dict didn't shadow
