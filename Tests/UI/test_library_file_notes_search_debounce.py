"""The file-notes search box debounces its execution and pre-folds entries."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    FILE_NOTES_SEARCH_DEBOUNCE_SECONDS,
    LibraryFileNotesWorkspace,
)

pytestmark = pytest.mark.bootstrap_profile


class _FakeTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _SearchWorkspace(LibraryFileNotesWorkspace):
    """Bare shell exposing only the search-handler collaborators."""

    def __init__(self) -> None:  # skip the heavy real widget init
        pass


def _workspace():
    workspace = _SearchWorkspace()
    workspace._search_query = ""
    workspace._search_generation = 0
    workspace._navigator_mode = "files"
    workspace._navigator_mode_before_git = "files"
    workspace._search_debounce_timer = None
    timers = []

    def set_timer(interval, callback):
        timer = _FakeTimer()
        timers.append((interval, callback, timer))
        return timer

    workspace.set_timer = set_timer
    begins = []

    def begin(query):
        begins.append(query)

    workspace._begin_search = begin
    workspace._sync_navigator_mode = lambda: None
    resets = []
    workspace._navigator_surface_refs = lambda: SimpleNamespace(
        results=SimpleNamespace(reset=lambda text: resets.append(text))
    )
    return workspace, timers, begins, resets


def _changed(value: str):
    from textual.widgets import Input

    return Input.Changed(input=None, value=value)  # type: ignore[arg-type]


def test_nonempty_change_debounces_instead_of_searching():
    workspace, timers, begins, _resets = _workspace()
    workspace._search_changed(_changed("read"))
    assert len(timers) == 1 and not begins
    assert timers[0][0] == FILE_NOTES_SEARCH_DEBOUNCE_SECONDS
    timers[0][1]()  # fire
    assert begins == ["read"]


def test_rapid_changes_cancel_earlier_timers():
    workspace, timers, begins, _resets = _workspace()
    workspace._search_changed(_changed("r"))
    workspace._search_changed(_changed("re"))
    workspace._search_changed(_changed("rea"))
    assert len(timers) == 3
    assert timers[0][2].stopped and timers[1][2].stopped
    assert not timers[2][2].stopped
    timers[2][1]()
    assert begins == ["rea"], "only the latest query runs"


def test_clearing_the_box_restores_files_mode_immediately():
    workspace, timers, begins, resets = _workspace()
    workspace._navigator_mode = "search"
    workspace._search_changed(_changed(""))
    assert workspace._navigator_mode == "files"
    assert resets and not timers and not begins


def test_submit_bypasses_the_debounce():
    from textual.widgets import Input

    workspace, timers, begins, _resets = _workspace()
    workspace._search_submitted(Input.Submitted(input=None, value="now"))  # type: ignore[arg-type]
    assert begins == ["now"] and not timers


@pytest.mark.asyncio
async def test_replica_less_fallback_matches_prefolded_paths():
    workspace = _SearchWorkspace()
    workspace._active = True
    workspace._replica = None
    workspace._service = None
    workspace._search_generation = 1
    workspace._entries = {"Docs/Readme.md": None, "SRC/Main.PY": None}
    workspace._entries_folded = tuple(
        path.casefold() for path in workspace._entries
    )
    rebuilt = []
    workspace._rebuild_search_results = lambda paths: rebuilt.append(paths)

    await workspace._run_search("read", generation=1)
    assert rebuilt == [("Docs/Readme.md",)]
    await workspace._run_search(".py", generation=1)
    assert rebuilt[-1] == ("SRC/Main.PY",), "folded matching is case-insensitive"
