from types import SimpleNamespace

from tldw_chatbook.UI.Screens.writing_screen import WritingScreen
from tldw_chatbook.UI.Writing_Window import WritingWindow


def test_writing_screen_composes_writing_window():
    app = SimpleNamespace(writing_scope_service=object())
    screen = WritingScreen(app)

    widgets = list(screen.compose_content())

    assert len(widgets) == 1
    assert isinstance(widgets[0], WritingWindow)


def test_writing_screen_round_trips_window_state():
    app = SimpleNamespace(writing_scope_service=object())
    screen = WritingScreen(app)
    window = WritingWindow(app)
    window.restore_state({"source": "server"})
    screen.query_one = lambda *_args, **_kwargs: window

    state = screen.save_state()
    screen.restore_state({"source": "local"})

    assert state == {"source": "server"}
    assert window.save_state() == {"source": "local"}
