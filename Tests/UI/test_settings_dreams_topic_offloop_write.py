"""Dreams topic writes run off the event loop, pane refresh on completion."""

import threading
from types import SimpleNamespace

import pytest
from textual.css.query import QueryError
from textual.widgets import Input, Select

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

pytestmark = pytest.mark.bootstrap_profile


class _DreamsDB:
    def __init__(self) -> None:
        self.write_threads: list[int] = []

    def upsert_profile_entry(self, *args, **kwargs):
        self.write_threads.append(threading.get_ident())

    def delete_profile_entry(self, entry_id):
        self.write_threads.append(threading.get_ident())


class _DreamsScreen(SettingsScreen):
    """Bare shell: only the topic handlers' collaborators are stubbed."""

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen(db):
    screen = _DreamsScreen()
    screen._settings_dreams_result = ""
    statuses = []
    screen._set_static_text = lambda selector, text: statuses.append(text)
    refreshes = []
    screen._refresh_dreams_pane = lambda: refreshes.append(None)
    screen._dreams_db = lambda: db
    workers = []

    def run_worker(coroutine, **kwargs):
        workers.append((coroutine, kwargs))

    screen.run_worker = run_worker

    def query_one(selector, cls=None):
        if selector == "#settings-dreams-topic-text":
            return SimpleNamespace(value="  quiet reading  ")
        if selector == "#settings-dreams-topic-weight":
            raise QueryError("default weight")
        raise QueryError(selector)

    screen.query_one = query_one
    return screen, workers, statuses, refreshes


@pytest.mark.asyncio
async def test_topic_add_writes_off_loop_then_refreshes():
    db = _DreamsDB()
    screen, workers, statuses, refreshes = _screen(db)

    event = SimpleNamespace(stop=lambda: None)
    screen.handle_dreams_topic_add(event)

    assert len(workers) == 1
    coroutine, kwargs = workers[0]
    assert kwargs.get("group") == "settings-dreams-topic-write"
    await coroutine()

    assert db.write_threads and all(
        thread != threading.get_ident() for thread in db.write_threads
    ), "the write must run off the event-loop thread"
    assert not statuses  # success path surfaces no inline error
    assert refreshes, "pane refresh applies on completion"
    assert "Added" in screen._settings_dreams_result


@pytest.mark.asyncio
async def test_topic_remove_writes_off_loop_then_refreshes():
    db = _DreamsDB()
    screen, workers, _statuses, refreshes = _screen(db)

    button = SimpleNamespace(
        id="settings-dreams-topic-remove-12", dreams_topic_entry_id="12"
    )
    event = SimpleNamespace(button=button, stop=lambda: None)
    screen.handle_dreams_topic_remove_pressed(event)

    assert len(workers) == 1
    await workers[0][0]()
    assert db.write_threads and all(
        thread != threading.get_ident() for thread in db.write_threads
    )
    assert refreshes
    assert screen._settings_dreams_result == "Topic removed."
