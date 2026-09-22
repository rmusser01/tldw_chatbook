"""The workflow picker must not search the database on every keystroke.

TASK-32901 (tier-2 S26 P2): ``PagedChoiceModal.search_page`` called
``load_page`` directly from ``Input.Changed``. ``load_page`` is
``@work(exclusive=True)``, but ``exclusive`` cancels the awaiting Textual
worker, not the thread already dispatched inside ``asyncio.to_thread`` -- so
an eight-character query queued eight unbounded ``workflow_heads`` scans.
The repo's own convention for this is a restart-on-keystroke timer
(``UI/Chatbooks_Window_Improved.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.UI.Workflows_Modules.library import PagedChoiceModal


class _Timer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _Stub:
    """Minimal stand-in exposing only what ``search_page`` touches."""

    SEARCH_DEBOUNCE_SECONDS = PagedChoiceModal.SEARCH_DEBOUNCE_SECONDS

    def __init__(self) -> None:
        self.loads: list[tuple[int, str]] = []
        self.timers: list[_Timer] = []
        self.callbacks: list = []
        self._search_timer = None

    def set_timer(self, delay, callback):
        assert delay > 0
        timer = _Timer()
        self.timers.append(timer)
        self.callbacks.append(callback)
        return timer

    def load_page(self, offset, query):
        self.loads.append((offset, query))


def _keystroke(value: str):
    return SimpleNamespace(value=value, stop=lambda: None)


def test_search_keystroke_defers_the_query_instead_of_loading_a_page():
    stub = _Stub()

    PagedChoiceModal.search_page(stub, _keystroke("b"))

    assert stub.loads == []
    assert len(stub.timers) == 1

    stub.callbacks[-1]()
    assert stub.loads == [(0, "b")]


def test_a_following_keystroke_cancels_the_pending_search():
    stub = _Stub()

    PagedChoiceModal.search_page(stub, _keystroke("b"))
    PagedChoiceModal.search_page(stub, _keystroke("bi"))

    assert stub.timers[0].stopped is True
    assert stub.loads == []

    stub.callbacks[-1]()
    assert stub.loads == [(0, "bi")]
