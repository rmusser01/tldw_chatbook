"""Meetings caches the per-event widget refs (ASR partial stream)."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.meetings_screen import MeetingsScreen

pytestmark = pytest.mark.bootstrap_profile


class _Widget:
    def __init__(self) -> None:
        self.is_attached = True
        self.renderable = ""
        self.updates: list[object] = []
        self.writes: list[object] = []

    def update(self, value):
        self.updates.append(value)

    def write(self, value):
        self.writes.append(value)


class _MeetingsShell(MeetingsScreen):
    """Bare shell exposing only the cached-ref collaborators."""

    is_mounted = True  # shadow the real property for the bare shell

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen():
    screen = _MeetingsShell()
    screen._partial_static = None
    screen._transcript_log = None
    screen._transcribing = False
    queries = []
    partials = [_Widget()]
    transcripts = [_Widget()]

    def query_one(selector, cls=None):
        queries.append(selector)
        if selector == "#meetings-partial":
            if len(queries) > len(partials):
                partials.append(_Widget())
            return partials[-1]
        if len(queries) > len(transcripts):
            transcripts.append(_Widget())
        return transcripts[-1]

    screen.query_one = query_one
    return screen, queries, partials[0], transcripts[0]


def test_partial_ref_is_cached_across_events():
    screen, queries, partial, _transcript = _screen()

    screen._partial_widget().update("one")
    screen._partial_widget().update("two")
    screen._partial_widget().update("three")

    assert len(queries) == 1, "one DOM walk, then cached ref reuse"
    assert partial.updates[-3:] == ["one", "two", "three"]


def test_detached_ref_is_requeried():
    screen, queries, partial, _transcript = _screen()
    screen._partial_widget().update("first")
    partial.is_attached = False  # a rebuild detached the cached widget

    screen._partial_widget().update("second")
    assert len(queries) == 2, "staleness re-queries once"
    assert screen._partial_static is not partial


def test_partial_event_uses_the_cached_ref():
    screen, queries, partial, _transcript = _screen()
    screen._coarse_label = lambda label: label
    screen._apply_event("partial", ("transcribed text", None))
    screen._apply_event("partial", ("more text", "S1"))
    screen._apply_event("transcribing", False)

    assert len(queries) == 1, "ASR events reuse the cached ref"
    # transcribing(False) is a no-op while the line already shows text.
    assert partial.updates == ["transcribed text…", "S1: more text…"]
