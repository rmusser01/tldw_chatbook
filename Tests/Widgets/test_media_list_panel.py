from unittest.mock import Mock

from tldw_chatbook.Widgets.Media.media_list_panel import (
    MediaItemSelectedEvent,
    MediaListPanel,
)


def test_build_selection_event_uses_normalized_record_id_and_preserves_media_data():
    panel = MediaListPanel(Mock())
    panel.call_later = Mock()

    panel.load_items(
        [
            {
                "id": "server:reading_item:118",
                "title": "Remote Article",
                "backing_media_id": 42,
                "backend": "server",
            }
        ],
        page=1,
        total_pages=1,
    )

    event = panel._build_selection_event_for_test(0)

    assert isinstance(event, MediaItemSelectedEvent)
    assert event.record_id == "server:reading_item:118"
    assert event.media_data["backing_media_id"] == 42
    assert event.media_data["backend"] == "server"


def test_row_widget_ids_are_index_based_not_record_id_based():
    panel = MediaListPanel(Mock())

    assert panel._row_widget_id(0) == "media-row-0"
    assert panel._row_widget_id(7) == "media-row-7"
