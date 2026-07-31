from types import SimpleNamespace
from unittest.mock import Mock

from tldw_chatbook.Widgets.Persona_Widgets.world_book_picker import WorldBookPicker


class _Listing:
    def __init__(self) -> None:
        self.rows = []
        self.index = None

    def clear(self) -> None:
        self.rows.clear()

    def append(self, row) -> None:
        self.rows.append(row)


def _picker(books):
    picker = WorldBookPicker(books)
    listing = _Listing()
    picker.query_one = Mock(return_value=listing)
    picker.dismiss = Mock()
    return picker, listing


def test_pick_returns_int_id():
    books = [
        {"world_book_id": 10, "name": "Alpha"},
        {"world_book_id": 20, "name": "Beta"},
    ]
    picker, listing = _picker(books)
    picker._populate(books)
    listing.index = 1
    event = SimpleNamespace(stop=Mock())

    picker._confirm(event)

    event.stop.assert_called_once_with()
    picker.dismiss.assert_called_once_with(20)


def test_filter_then_select():
    books = [
        {"world_book_id": 10, "name": "Alpha"},
        {"world_book_id": 20, "name": "Beta"},
    ]
    picker, listing = _picker(books)

    picker._filter(SimpleNamespace(value="beta", stop=Mock()))
    listing.index = 0

    assert picker._selected_id() == 20
    assert picker._row_ids == [20]


def test_cancel_returns_none():
    picker, _listing = _picker([{"world_book_id": 10, "name": "Alpha"}])
    event = SimpleNamespace(stop=Mock())

    picker._cancel(event)

    event.stop.assert_called_once_with()
    picker.dismiss.assert_called_once_with(None)
