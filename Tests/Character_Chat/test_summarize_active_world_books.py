import pytest

from tldw_chatbook.Character_Chat.world_info_resolver import (
    summarize_active_world_books,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "summarize_wb.db", "test-client")
    yield db
    db.close_connection()


def test_lists_attached_books_with_counts(wb_db):
    wb_db.add_conversation({"id": "c1", "title": "C"})
    wb = WorldBookManager(wb_db)
    b1 = wb.create_world_book("Alpha")
    wb.create_world_book_entry(b1, keys=["a"], content="x")
    wb.create_world_book_entry(b1, keys=["b"], content="y")
    wb.associate_world_book_with_conversation("c1", b1)
    out = summarize_active_world_books(wb_db, "c1", None)
    assert out["world_books"] == [{"name": "Alpha", "enabled": True, "entry_count": 2}]


def test_includes_disabled_attached_book(wb_db):
    wb_db.add_conversation({"id": "c2", "title": "C"})
    wb = WorldBookManager(wb_db)
    b = wb.create_world_book("Off", enabled=False)
    wb.create_world_book_entry(b, keys=["k"], content="x")
    wb.associate_world_book_with_conversation("c2", b)
    out = summarize_active_world_books(wb_db, "c2", None)
    assert out["world_books"] == [{"name": "Off", "enabled": False, "entry_count": 1}]


def test_empty_when_none_attached(wb_db):
    wb_db.add_conversation({"id": "c3", "title": "C"})
    assert summarize_active_world_books(wb_db, "c3", None) == {"world_books": [], "source": "local"}


def test_no_conversation_returns_empty(wb_db):
    assert summarize_active_world_books(wb_db, None, None) == {"world_books": [], "source": "local"}


def test_db_error_returns_empty():
    assert summarize_active_world_books(object(), "cX", None) == {"world_books": [], "source": "local"}
