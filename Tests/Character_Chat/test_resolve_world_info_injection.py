import pytest

from tldw_chatbook.Character_Chat.world_info_resolver import (
    resolve_world_info_injection,
    apply_world_info_to_message,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "resolve_wi.db", "test-client")
    yield db
    db.close_connection()


def _attach(db, conv_id, key, content, name="Lore"):
    db.add_conversation({"id": conv_id, "title": "C"})
    wb = WorldBookManager(db)
    book_id = wb.create_world_book(name)
    wb.create_world_book_entry(book_id, keys=[key], content=content)
    wb.associate_world_book_with_conversation(conv_id, book_id)
    return book_id


def test_returns_text_and_count_on_match(wb_db):
    _attach(wb_db, "c1", "dragon", "Dragons breathe fire.")
    text, count = resolve_world_info_injection(wb_db, "c1", None, "a dragon appears", [])
    assert "Dragons breathe fire." in text and count == 1


def test_conversation_book_applies_without_character(wb_db):
    # The gate-fix behavior: char_data=None (no character) still injects.
    _attach(wb_db, "c2", "griffin", "Griffins soar.")
    text, count = resolve_world_info_injection(wb_db, "c2", None, "a griffin flies", [])
    assert "Griffins soar." in text and count == 1


def test_no_match_returns_unchanged_zero(wb_db):
    _attach(wb_db, "c3", "dragon", "x")
    assert resolve_world_info_injection(wb_db, "c3", None, "hello", []) == ("hello", 0)


def test_no_conversation_and_db_error_zero(wb_db):
    assert resolve_world_info_injection(wb_db, None, None, "a dragon appears", []) == ("a dragon appears", 0)
    assert resolve_world_info_injection(object(), "cX", None, "a dragon appears", []) == ("a dragon appears", 0)


def test_apply_wrapper_returns_only_text(wb_db):
    _attach(wb_db, "c4", "dragon", "Dragons breathe fire.")
    text = apply_world_info_to_message(wb_db, "c4", None, "a dragon appears", [])
    text2, _ = resolve_world_info_injection(wb_db, "c4", None, "a dragon appears", [])
    assert text == text2 and isinstance(text, str)
