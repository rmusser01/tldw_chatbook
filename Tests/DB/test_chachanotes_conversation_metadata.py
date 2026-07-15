"""P1e: conversation metadata must be writable via update_conversation."""

import json

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield database
    database.close_connection()


def _new_conversation(db) -> str:
    return db.add_conversation({"title": "Chat A"})


def test_update_conversation_persists_metadata_and_bumps_version(db):
    conv_id = _new_conversation(db)
    before = db.get_conversation_by_id(conv_id)
    blob = json.dumps({"active_dictionaries": [3, 7]})

    ok = db.update_conversation(conv_id, {"metadata": blob}, expected_version=before["version"])
    assert ok

    after = db.get_conversation_by_id(conv_id)
    assert json.loads(after["metadata"]) == {"active_dictionaries": [3, 7]}
    assert after["version"] == before["version"] + 1


def test_title_only_update_still_works(db):
    conv_id = _new_conversation(db)
    v = db.get_conversation_by_id(conv_id)["version"]
    db.update_conversation(conv_id, {"title": "Renamed"}, expected_version=v)
    assert db.get_conversation_by_id(conv_id)["title"] == "Renamed"
