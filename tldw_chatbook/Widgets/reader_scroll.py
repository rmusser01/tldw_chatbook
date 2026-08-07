"""Reader-key scroll container for read-only markdown panes (TASK-1994).

Terminal-native document keys, ported from frogmouth's viewer bindings:
``j``/``k`` scroll by line, ``space``/``b`` by page. Used by the HF README
pane and the media content/analysis viewers. The Console transcript is
deliberately NOT one of these — its ``j``/``k`` select messages.
"""

from textual.binding import Binding
from textual.containers import VerticalScroll


class ReaderVerticalScroll(VerticalScroll):
    """VerticalScroll with j/k line and space/b page scrolling when focused.

    The bindings surface in the footer key hints while the pane has focus
    (the app's discoverability convention); arrow/PageUp/PageDown keys keep
    working via the base class.
    """

    BINDINGS = [
        Binding("j", "scroll_down", "Line ↓", show=True),
        Binding("k", "scroll_up", "Line ↑", show=True),
        Binding("space", "page_down", "Page ↓", show=True),
        Binding("b", "page_up", "Page ↑", show=True),
    ]
