import pytest

from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "apply_chatdicts.db", "test-client")
    yield database
    database.close_connection()


def _attach_matching_dict(db, conv_id, name="Slang", key="Warden", content="grim jailer"):
    dict_id = cdl.save_chat_dictionary(
        db, name, entries=[cdl.ChatDictionary(key=key, content=content)]
    )
    assert dict_id is not None
    LocalChatDictionaryService(db).attach_to_conversation(dict_id, conv_id)
    return dict_id


def test_applies_attached_conversation_dictionary(db):
    conv_id = db.add_conversation({"title": "Attach"})
    _attach_matching_dict(db, conv_id)
    out = cdl.apply_active_chatdicts_to_text(
        db, conv_id, None, "The Warden nods.", max_tokens=500, strategy="sorted_evenly"
    )
    assert out == "The grim jailer nods."


def test_no_conversation_returns_text_unchanged(db):
    out = cdl.apply_active_chatdicts_to_text(
        db, None, None, "The Warden nods.", max_tokens=500, strategy="sorted_evenly"
    )
    assert out == "The Warden nods."


def test_conversation_without_dicts_returns_text_unchanged(db):
    conv_id = db.add_conversation({"title": "Empty"})
    out = cdl.apply_active_chatdicts_to_text(
        db, conv_id, None, "The Warden nods.", max_tokens=500, strategy="sorted_evenly"
    )
    assert out == "The Warden nods."


def test_non_string_text_returned_unchanged(db):
    conv_id = db.add_conversation({"title": "T"})
    _attach_matching_dict(db, conv_id)
    sentinel = ["not", "a", "string"]
    assert cdl.apply_active_chatdicts_to_text(db, conv_id, None, sentinel) is sentinel


def test_never_raises_when_collect_fails(db, monkeypatch):
    conv_id = db.add_conversation({"title": "T"})
    _attach_matching_dict(db, conv_id)

    def _boom(*a, **k):
        raise RuntimeError("collect exploded")

    monkeypatch.setattr(cdl, "collect_active_chatdict_entries", _boom)
    out = cdl.apply_active_chatdicts_to_text(db, conv_id, None, "The Warden nods.")
    assert out == "The Warden nods."
