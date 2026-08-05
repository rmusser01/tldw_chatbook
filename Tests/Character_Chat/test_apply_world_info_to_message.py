import pytest

from tldw_chatbook.Character_Chat.world_info_resolver import (
    apply_world_info_to_message,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "world_info_resolver.db", "test-client")
    yield db
    db.close_connection()


def _attach_book(db, conv_id, key, content, title="Case"):
    db.add_conversation({"id": conv_id, "title": title})
    wb = WorldBookManager(db)
    book_id = wb.create_world_book("Lore")
    wb.create_world_book_entry(book_id, keys=[key], content=content)
    wb.associate_world_book_with_conversation(conv_id, book_id)
    return book_id


def test_injects_matched_conversation_book(wb_db):
    _attach_book(wb_db, "conv-1", "dragon", "Dragons breathe fire.")
    out = apply_world_info_to_message(wb_db, "conv-1", None, "a dragon appears", [])
    assert "Dragons breathe fire." in out
    assert out.endswith("a dragon appears") or "a dragon appears" in out
    assert out != "a dragon appears"


def test_no_match_returns_unchanged(wb_db):
    _attach_book(wb_db, "conv-2", "dragon", "Dragons breathe fire.")
    assert apply_world_info_to_message(wb_db, "conv-2", None, "hello there", []) == "hello there"


def test_no_conversation_returns_unchanged(wb_db):
    assert apply_world_info_to_message(wb_db, None, None, "a dragon appears", []) == "a dragon appears"


def test_no_books_returns_unchanged(wb_db):
    wb_db.add_conversation({"id": "conv-3", "title": "Empty"})
    assert apply_world_info_to_message(wb_db, "conv-3", None, "a dragon appears", []) == "a dragon appears"


def test_db_error_returns_unchanged():
    # A bogus db object: the helper must swallow the error and return the text.
    assert apply_world_info_to_message(object(), "conv-x", None, "a dragon appears", []) == "a dragon appears"


def test_non_string_message_returned_as_is(wb_db):
    assert apply_world_info_to_message(wb_db, "conv-1", None, None, []) is None


def test_character_only_book_not_injected_when_char_data_none(wb_db):
    # A book attached to a CHARACTER (char_data=None here) must not inject.
    char_id = wb_db.add_character_card({"name": "Hero"})
    wb = WorldBookManager(wb_db)
    book_id = wb.create_world_book("CharLore")
    wb.create_world_book_entry(book_id, keys=["griffin"], content="Griffins soar.")
    wb.attach_world_book_to_character(book_id, char_id)
    wb_db.add_conversation({"id": "conv-4", "title": "NoConvBook"})
    out = apply_world_info_to_message(wb_db, "conv-4", None, "a griffin flies", [])
    assert out == "a griffin flies"  # char_data is None → character books never apply
