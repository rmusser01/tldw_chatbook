import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    # File-backed (NOT :memory:) — CharactersRAGDB uses threading.local connections
    # and the production avatar fetch runs off-thread; :memory: is connection-private.
    return CharactersRAGDB(tmp_path / "expr.db", "test-client")


def _make_character(db) -> int:
    return db.add_character_card({"name": "Ada"})


def test_schema_is_v23(db):
    assert db._get_db_version(db.get_connection()) == 23


def test_set_then_get_round_trips(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"PNGBYTES", "image/png")
    assert db.get_character_expression_image(cid, "speaking") == b"PNGBYTES"


def test_get_missing_returns_none(db):
    cid = _make_character(db)
    assert db.get_character_expression_image(cid, "thinking") is None


def test_set_is_upsert_on_character_and_state(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"one")
    db.set_character_expression_image(cid, "speaking", b"two")
    assert db.get_character_expression_image(cid, "speaking") == b"two"
    assert db.list_character_expression_states(cid) == ["speaking"]  # not duplicated


def test_list_states_returns_only_active(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "thinking", b"t")
    db.set_character_expression_image(cid, "error", b"e")
    assert sorted(db.list_character_expression_states(cid)) == ["error", "thinking"]


def test_delete_is_soft_and_get_returns_none(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"s")
    db.delete_character_expression_image(cid, "speaking")
    assert db.get_character_expression_image(cid, "speaking") is None
    assert db.list_character_expression_states(cid) == []


def test_set_after_delete_reactivates(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"s1")
    db.delete_character_expression_image(cid, "speaking")
    db.set_character_expression_image(cid, "speaking", b"s2")
    assert db.get_character_expression_image(cid, "speaking") == b"s2"


def test_invalid_state_id_rejected(db):
    cid = _make_character(db)
    with pytest.raises(ValueError):
        db.set_character_expression_image(cid, "idle", b"x")   # idle is never stored
    with pytest.raises(ValueError):
        db.set_character_expression_image(cid, "bogus", b"x")
