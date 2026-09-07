"""Direct function tests for the TagFilterPicker modal."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.Widgets.Persona_Widgets.tag_filter_picker import TagFilterPicker


class _Listing:
    def __init__(self) -> None:
        self.rows = []
        self.index = None

    def clear(self) -> None:
        self.rows.clear()

    def append(self, row) -> None:
        self.rows.append(row)


def _picker(tags: list[str], current: str | None = None):
    picker = TagFilterPicker(tags, current)
    listing = _Listing()
    picker.query_one = Mock(return_value=listing)
    picker.dismiss = Mock()
    return picker, listing


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, None), (1, "hero"), (2, "hero mage")],
)
def test_selected_row_returns_stored_value(index, expected):
    picker, listing = _picker(["hero", "hero mage"])
    picker._populate(picker._tags)
    listing.index = index
    event = SimpleNamespace(stop=Mock())

    picker._selected(event)

    event.stop.assert_called_once_with()
    picker.dismiss.assert_called_once_with(expected)


def test_search_narrows_rows_and_preserves_exact_tag_mapping():
    picker, listing = _picker(["alpha", "beta", "gamma"])

    # task-15476: `_filter` itself now only arms a debounce `Timer`, which
    # requires a running Textual message pump this synchronous, unmounted
    # unit test never provides. `_apply_filter_debounced` is where the
    # actual filter-then-populate logic this test exercises now lives; the
    # debounce mechanics themselves are covered by a dedicated test that
    # runs a real App (`test_console_character_picker_debounce.py`).
    picker._apply_filter_debounced("beta")

    assert len(listing.rows) == 2
    assert picker._row_tags == [None, "beta"]


def test_escape_cancels_distinct_from_clear_filter():
    picker, _listing = _picker(["hero"])
    event = SimpleNamespace(key="escape", stop=Mock())

    picker.on_key(event)

    event.stop.assert_called_once_with()
    picker.dismiss.assert_called_once_with(TagFilterPicker.CANCEL)
    assert TagFilterPicker.CANCEL is not None
