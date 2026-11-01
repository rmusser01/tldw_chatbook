"""Library must paint before preparing its app-owned sync runtime."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.screen import Screen
from textual.widgets import Static

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    InertLastingSyncRuntime,
    LibraryNotesSyncController,
)

pytestmark = pytest.mark.bootstrap_profile


def test_screen_constructor_does_not_build_the_lazy_owner(monkeypatch):
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    reader = Mock(return_value=InertLastingSyncRuntime())
    monkeypatch.setattr(app, "_read_notes_sync_runtime_owner_safe", reader)
    LibraryScreen(app)
    reader.assert_not_called()


class PreparationScreen(Screen):
    _run_library_service_call = staticmethod(LibraryScreen._run_library_service_call)

    def __init__(self, app_owner):
        super().__init__()
        self.app_instance = app_owner
        self._notes_sync_runtime_prepared = False
        self._notes_sync_runtime_preparing = False
        self._library_notes_sync_controller = LibraryNotesSyncController(
            runtime=InertLastingSyncRuntime(status="starting", next_action="wait"),
            import_controller=SimpleNamespace(),
        )

    def compose(self):
        yield Static("Library")

    def on_mount(self):
        self.run_worker(LibraryScreen._prepare_notes_sync_runtime(self))

    def refresh_notes_sync_runtime(self):
        self._library_notes_sync_controller.refresh_roots()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["ready", "failed", "unmount"])
async def test_runtime_preparation_does_not_block_or_publish_after_unmount(outcome):
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    observed = []
    runtime = InertLastingSyncRuntime(status="not_configured")
    owner = SimpleNamespace(
        ensure_notes_sync_runtime_started=Mock(),
        _notes_sync_runtime_owner=None,
        _notes_sync_runtime_build_failure=None,
    )

    def read():
        observed.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        if outcome == "failed":
            owner._notes_sync_runtime_build_failure = ValueError("synthetic failure")
            return None
        return runtime

    owner._read_notes_sync_runtime_owner_safe = read
    owner.read_notes_sync_runtime_owner = read
    owner.peek_notes_sync_runtime = lambda: (
        owner._notes_sync_runtime_owner,
        owner._notes_sync_runtime_build_failure is not None,
    )
    screen = PreparationScreen(owner)
    app = App()
    try:
        async with app.run_test() as pilot:
            await app.push_screen(screen)
            async with asyncio.timeout(3):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            assert observed[0] != loop_thread
            assert not screen._library_notes_sync_controller.snapshot.lasting_available
            if outcome == "unmount":
                await app.pop_screen()
            release.set()
            if outcome != "unmount":
                async with asyncio.timeout(3):
                    while not screen._notes_sync_runtime_prepared:
                        await asyncio.sleep(0.01)
                controller = screen._library_notes_sync_controller
                assert controller.snapshot.lasting_available == (outcome == "ready")
                if outcome == "ready":
                    owner.ensure_notes_sync_runtime_started.assert_called_once_with()
                else:
                    owner.ensure_notes_sync_runtime_started.assert_not_called()
                    assert "could not start" in controller._unavailable_copy()
            else:
                await pilot.pause(0.1)
                assert not screen._notes_sync_runtime_prepared
                owner.ensure_notes_sync_runtime_started.assert_not_called()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_notes_visit_attention_does_not_build_owner_on_ui_loop():
    from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention

    loop_thread = threading.get_ident()
    observed = []

    class Owner:
        _notes_sync_runtime_owner = None

        def __init__(self):
            self._notes_sync_runtime_owner = None

        @property
        def notes_sync_runtime_owner(self):
            observed.append(threading.get_ident())
            assert threading.get_ident() != loop_thread
            self._notes_sync_runtime_owner = InertLastingSyncRuntime()
            return self._notes_sync_runtime_owner

        def peek_notes_sync_runtime(self):
            return self._notes_sync_runtime_owner, False

        def read_notes_sync_runtime_owner(self):
            return self.notes_sync_runtime_owner

    host = SimpleNamespace(
        app_instance=Owner(),
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _notes_state=SimpleNamespace(),
    )
    await library_notes_sync_attention.refresh_library_notes_sync_attention(host)
    assert observed and observed[0] != loop_thread


@pytest.mark.asyncio
async def test_failed_runtime_preparation_keeps_known_attention():
    from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention

    owner = SimpleNamespace(
        _notes_sync_runtime_owner=None,
        _notes_sync_runtime_build_failure=ValueError("synthetic failure"),
        read_notes_sync_runtime_owner=lambda: None,
        peek_notes_sync_runtime=lambda: (None, True),
    )
    notes_state = SimpleNamespace(tree_attention_folder_ids=frozenset({"held-folder"}))
    host = SimpleNamespace(
        app_instance=owner,
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _notes_state=notes_state,
    )
    await library_notes_sync_attention.refresh_library_notes_sync_attention(host)
    assert notes_state.tree_attention_folder_ids == frozenset({"held-folder"})
