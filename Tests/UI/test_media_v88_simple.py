"""Direct compatibility-export contracts for the retired V88 module name."""

from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.MediaWindowV88 import (
    MediaItemSelectedEventV88,
    MediaSearchEventV88,
    MediaTypeSelectedEventV88,
    MediaWindowV88,
)
from tldw_chatbook.Widgets.Media import MediaItemSelectedEvent, MediaSearchEvent
from tldw_chatbook.Widgets.Media.media_navigation_panel import MediaTypeSelectedEvent


def test_media_window_v88_exports_are_direct_aliases():
    assert MediaWindowV88 is MediaWindow
    assert MediaItemSelectedEventV88 is MediaItemSelectedEvent
    assert MediaSearchEventV88 is MediaSearchEvent
    assert MediaTypeSelectedEventV88 is MediaTypeSelectedEvent
