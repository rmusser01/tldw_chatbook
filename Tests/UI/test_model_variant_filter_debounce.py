"""The model-browser variant filter debounces its row rebuild."""

from types import SimpleNamespace

import pytest
from textual.widgets import Input

from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

pytestmark = pytest.mark.bootstrap_profile


class _FilterView(RemoteView):
    """Bare shell: only the filter-handler collaborators are stubbed."""

    def __init__(self) -> None:  # skip the heavy real view init
        pass


class _FakeTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _view(resolved):
    view = _FilterView()
    view._resolved = resolved
    view._selected_candidate = None
    view._variant_filter_timer = None
    timers = []

    def set_timer(interval, callback):
        timer = _FakeTimer()
        timers.append((interval, callback, timer))
        return timer

    view.set_timer = set_timer
    rebuilds = []
    view._refresh_variant_list = lambda: rebuilds.append(None)
    view._variant_guidance_rows = lambda resolved: []
    view._set_status = lambda message: None
    view._selection_summary = lambda: ""
    view._stop_variant_filter_timer_call = None
    return view, timers, rebuilds


def _changed(value: str):
    return Input.Changed(input=None, value=value)  # type: ignore[arg-type]


def test_filter_change_debounces_the_rebuild():
    view, timers, rebuilds = _view(SimpleNamespace(candidates=[]))
    view._variant_filter_changed(_changed("q4"))
    assert len(timers) == 1 and timers[0][0] == 0.2 and not rebuilds
    timers[0][1]()
    assert rebuilds, "the rebuild runs once typing pauses"


def test_rapid_changes_cancel_earlier_timers():
    view, timers, rebuilds = _view(SimpleNamespace(candidates=[]))
    view._variant_filter_changed(_changed("q"))
    view._variant_filter_changed(_changed("q4"))
    view._variant_filter_changed(_changed("q4_0"))
    assert timers[0][2].stopped and timers[1][2].stopped
    assert not timers[2][2].stopped
    timers[2][1]()
    assert len(rebuilds) == 1


def test_unresolved_view_does_not_schedule():
    view, timers, rebuilds = _view(None)
    view._variant_filter_changed(_changed("q4"))
    assert not timers and not rebuilds
