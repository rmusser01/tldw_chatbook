"""A run tick's Inspector/rail context build reads no registry or appearance rows on the loop.

TASK-33620.15.4 (measured live 2026-10-23, base 5b5920b8e2): once per send the
commit's workspace-membership write moved the registry's mutation generation,
and the next run tick's projection re-read the active workspace, the
workspace list and the memberships synchronously (storage admission
included) inside the Inspector's and rail's shared context build; the
browser rows' appearance read did the same whenever its 2 s cache expired or
a new conversation appeared.

While the run poll ticks, a tick's build now shows the last snapshot and
starts one off-loop re-read, published to the next tick. These tests drive
the real Workspace controller and registry over a file-backed database and
pin: no registry or appearance SQL on the UI loop's thread inside such a
build; the re-read landing installs and publishes (freshness: the next build
reads it with no SQL); a re-read overtaken by a newer mutation, or by an
appearance write, installs nothing; with no run poll every read stays live.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_workspace_controller import (
    _AsyncWorkerScreen,
    _workspace_controller,
)
from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile


def _reader_codes() -> frozenset:
    """The registry's display readers and the appearance batch read."""
    names = ("get_active_workspace", "list_workspaces", "list_workspace_memberships")
    codes = {
        inspect.unwrap(getattr(LocalWorkspaceRegistryService, name)).__code__
        for name in names
    }
    codes.add(CharactersRAGDB.get_conversations_metadata_by_ids.__code__)
    codes.add(base_db._QuiescentSQLiteCursor.execute.__code__)
    return frozenset(codes)


class _Statements:
    """Display reads by thread (observed with a profiler, nothing replaced)."""

    def __init__(self) -> None:
        self._codes = _reader_codes()
        self.main = 0
        self.other = 0

    def __enter__(self):
        self._previous = sys.getprofile(), threading.getprofile()
        main = threading.main_thread().ident

        def observe(frame, event, _argument):
            if event == "call" and frame.f_code in self._codes:
                if threading.get_ident() == main:
                    self.main += 1
                else:
                    self.other += 1

        threading.setprofile_all_threads(observe)
        return self

    def __exit__(self, *_exc):
        threading.setprofile_all_threads(self._previous[1])
        sys.setprofile(self._previous[0])


class _RunScreen(_AsyncWorkerScreen):
    """A screen whose run poll ticks (its timer exists)."""

    def __init__(self) -> None:
        super().__init__()
        self._console_transcript_sync_timer = object()

    @property
    def poll_asks(self) -> int:
        """Publications handed to the poll's next tick (a cadence input)."""
        from tldw_chatbook.UI.Console_Modules import config_sync

        return getattr(config_sync._cadence(self), "published", 0)

    async def _sync_native_console_chat_ui(self) -> None:
        raise AssertionError("a landing started a whole sync while the poll ran")


def _stale_reads(workspace):
    """The tick build's stale-read scope (absent on the base: live reads)."""
    return getattr(workspace, "_stale_display_reads", nullcontext)()


async def _settle(screen) -> None:
    while True:
        pending = [task for task, _kwargs in screen.workers if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending)


@pytest.fixture
def registry(tmp_path):
    database = WorkspaceDB(tmp_path / "registry.sqlite", client_id="display-reads")
    service = LocalWorkspaceRegistryService(database)
    service.create_workspace(workspace_id="alpha", name="Alpha")
    yield service
    database.close()


@pytest.mark.asyncio
async def test_a_run_tick_build_reads_a_moved_registry_off_the_ui_loop(registry):
    screen = _RunScreen()
    workspace = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    view = workspace._console_registry_reads_view()
    before = view.list_workspaces()
    # The commit's registry write: the generation moves.
    registry.create_workspace(workspace_id="beta", name="Beta")
    try:
        with _Statements() as sql, _stale_reads(workspace):
            shown = view.list_workspaces()
        assert sql.main == 0, (
            f"the run tick's build ran {sql.main} registry statement(s) on the UI loop"
        )
        assert [item.workspace_id for item in shown] == [
            item.workspace_id for item in before
        ]
        with _Statements() as sql:
            await _settle(screen)
        assert sql.main == 0 and sql.other > 0
        # Freshness: the landing publishes to the next tick, which reads it
        # from the snapshot with no SQL.
        assert screen.poll_asks == 1
        with _Statements() as sql, _stale_reads(workspace):
            fresh = view.list_workspaces()
        assert sql.main == 0 and sql.other == 0
        assert "beta" in {item.workspace_id for item in fresh}
    finally:
        await _settle(screen)


@pytest.mark.asyncio
async def test_a_reread_overtaken_by_a_newer_mutation_installs_nothing(registry):
    screen = _RunScreen()
    workspace = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    view = workspace._console_registry_reads_view()
    view.list_workspaces()
    registry.create_workspace(workspace_id="beta", name="Beta")
    entered, release = threading.Event(), threading.Event()
    original = registry.list_workspaces

    def held(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    registry.list_workspaces = held
    try:
        with _stale_reads(workspace):
            view.list_workspaces()
        assert screen.workers, "the build started no off-loop re-read"
        assert await asyncio.to_thread(entered.wait, 5)
        registry.create_workspace(workspace_id="gamma", name="Gamma")
        release.set()
        await _settle(screen)
        assert screen.poll_asks == 0, "a re-read of an older generation published"
        # The newer generation is read (live here: no tick build) in full.
        assert "gamma" in {item.workspace_id for item in view.list_workspaces()}
    finally:
        release.set()
        await _settle(screen)


@pytest.mark.asyncio
async def test_without_a_run_poll_every_registry_read_stays_live(registry):
    screen = _RunScreen()
    screen._console_transcript_sync_timer = None
    workspace = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    view = workspace._console_registry_reads_view()
    view.list_workspaces()
    registry.create_workspace(workspace_id="beta", name="Beta")
    with _Statements() as sql, _stale_reads(workspace):
        shown = view.list_workspaces()
    assert sql.main > 0
    assert "beta" in {item.workspace_id for item in shown}
    assert not screen.workers


@pytest.mark.asyncio
async def test_a_run_tick_build_rereads_an_expired_appearance_map_off_the_ui_loop(
    tmp_path, registry
):
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_appearance import ConsoleConversationAppearance
    from tldw_chatbook.UI.Console_Modules import workspace as module

    chat_db = CharactersRAGDB(tmp_path / "chat.sqlite", "display-reads")
    try:
        conversation_id = chat_db.add_conversation({"title": "One"})
        service = ChatConversationService(chat_db)
        screen = _RunScreen()
        workspace = _workspace_controller(
            screen=screen,
            app_instance=SimpleNamespace(
                workspace_registry_service=registry,
                local_chat_conversation_service=service,
            ),
        )
        first = workspace._console_conversation_appearance_map([conversation_id])
        assert conversation_id in first
        cached = workspace._console_appearance_map_cache
        # Another writer (not this controller's write path, which clears the
        # map) changes the row, so the re-read has something to publish.
        assert service.set_conversation_appearance(
            conversation_id=conversation_id,
            appearance=ConsoleConversationAppearance(icon="🧪", color="#f87171"),
        )
        workspace._console_appearance_map_cached_at -= (
            module.CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS + 1
        )
        with _Statements() as sql, _stale_reads(workspace):
            shown = workspace._console_conversation_appearance_map([conversation_id])
        assert sql.main == 0, "the expired appearance map was re-read on the UI loop"
        assert shown is cached
        await _settle(screen)
        assert screen.poll_asks == 1
        assert workspace._console_appearance_map_cache is not cached
        assert workspace._console_appearance_map_cache[conversation_id] == (
            "🧪",
            "#f87171",
        )

        # An unchanged re-read refreshes the map's age and publishes nothing:
        # a publication forces the next tick's projection (review 2026-10-25).
        unchanged = workspace._console_appearance_map_cache
        workspace._console_appearance_map_cached_at -= (
            module.CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS + 1
        )
        with _stale_reads(workspace):
            workspace._console_conversation_appearance_map([conversation_id])
        await _settle(screen)
        assert screen.poll_asks == 1, "an unchanged appearance re-read published"
        assert workspace._console_appearance_map_cache is unchanged
        started = len(screen.workers)
        with _stale_reads(workspace):
            workspace._console_conversation_appearance_map([conversation_id])
        assert len(screen.workers) == started, "a fresh map was re-read again"

        # A write clears the map while a re-read is in flight: it is dropped.
        workspace._console_appearance_map_cached_at -= (
            module.CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS + 1
        )
        with _stale_reads(workspace):
            workspace._console_conversation_appearance_map([conversation_id])
        workspace._console_appearance_map_cache = None  # The write's clear.
        await _settle(screen)
        assert workspace._console_appearance_map_cache is None
        assert screen.poll_asks == 1
    finally:
        chat_db.close()


@pytest.mark.asyncio
async def test_a_sends_start_reads_the_registry_live_until_its_provider_request(
    registry,
):
    """Measured live 2026-10-23: an off-loop re-read opens its own SQLite
    connection (34 ms, 10 ms CPU in isolation; 20-60 ms CPU in the app), and
    those reads, started right after a send's durable commit, delayed first
    sends by ~66 ms (8 sends per arm). From acceptance to the provider request
    the build reads live, as on the base (a 0.3-3 ms read on the loop)."""
    from tldw_chatbook.Chat import console_send_diagnostics as diagnostics
    from tldw_chatbook.UI.Console_Modules import config_sync

    screen = _RunScreen()
    workspace = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    view = workspace._console_registry_reads_view()
    view.list_workspaces()
    getattr(config_sync, "note_send_accepted", lambda _screen: None)(screen)
    registry.create_workspace(workspace_id="beta", name="Beta")
    try:
        with _Statements() as sql, _stale_reads(workspace):
            shown = view.list_workspaces()
        assert sql.main > 0, "a send's start re-read the registry off the UI loop"
        assert "beta" in {item.workspace_id for item in shown}
        assert not screen.workers

        # Once the send has made its provider request, reads go off-loop again.
        diagnostics.record_send_stage("provider_entry")
        registry.create_workspace(workspace_id="gamma", name="Gamma")
        with _Statements() as sql, _stale_reads(workspace):
            view.list_workspaces()
        assert sql.main == 0
        assert screen.workers
    finally:
        await _settle(screen)


@pytest.mark.asyncio
async def test_an_id_the_appearance_read_never_returns_does_not_loop_the_run_ticks(
    tmp_path, registry
):
    """Review 2026-10-25: Console delete soft-deletes a conversation but keeps
    its workspace membership, so its id still reaches the appearance read,
    which returns only live conversations. Every run-tick build then
    re-requested the map and every landing published (a projection input):
    one forced projection plus one connection-opening read per 0.2 s tick
    for the whole run. An unchanged landing publishes nothing, and an id a
    read found absent is not re-requested while the map is fresh."""
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    chat_db = CharactersRAGDB(tmp_path / "chat.sqlite", "display-reads")
    try:
        kept = chat_db.add_conversation({"title": "Kept"})
        gone = chat_db.add_conversation({"title": "Gone"})
        version = chat_db.get_conversation_by_id(gone)["version"]
        assert chat_db.soft_delete_conversation(gone, version)
        screen = _RunScreen()
        workspace = _workspace_controller(
            screen=screen,
            app_instance=SimpleNamespace(
                workspace_registry_service=registry,
                local_chat_conversation_service=ChatConversationService(chat_db),
            ),
        )
        wanted = [kept, gone]
        assert gone not in workspace._console_conversation_appearance_map(wanted)
        reads = 0
        for _tick in range(6):
            started = len(screen.workers)
            with _Statements() as sql, _stale_reads(workspace):
                shown = workspace._console_conversation_appearance_map(wanted)
            assert sql.main == 0, "a run tick's build read appearances on the UI loop"
            assert kept in shown
            reads += len(screen.workers) - started
            await _settle(screen)
        assert screen.poll_asks <= 1, (
            f"6 run-tick builds published {screen.poll_asks} times for a map "
            "whose rows never changed"
        )
        assert reads <= 1, f"6 run-tick builds started {reads} appearance re-reads"
    finally:
        chat_db.close()


@pytest.mark.asyncio
async def test_a_registry_read_that_keeps_failing_off_loop_publishes_once(registry):
    """Review 2026-10-25: one display read that always raises off-loop left its
    key stale, so every build re-requested the whole snapshot and every
    landing re-installed the reads that had already landed and published
    again -- the same self-sustaining tick loop. A landing that installs
    nothing new publishes nothing."""
    screen = _RunScreen()
    workspace = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    view = workspace._console_registry_reads_view()
    view.list_workspaces()
    view.list_workspace_memberships("alpha")
    registry.create_workspace(workspace_id="beta", name="Beta")
    original = registry.list_workspace_memberships

    def fails_off_loop(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("the memberships read keeps failing")
        return original(*args, **kwargs)

    registry.list_workspace_memberships = fails_off_loop
    try:
        for _tick in range(6):
            with _Statements() as sql, _stale_reads(workspace):
                shown = view.list_workspaces()
                view.list_workspace_memberships("alpha")
            assert sql.main == 0
            await _settle(screen)
        assert "beta" in {item.workspace_id for item in shown}
        assert screen.poll_asks == 1, (
            f"6 run-tick builds published {screen.poll_asks} times; only the "
            "first landing installed anything"
        )
        # Each guard on its own: a landed read is not re-requested, and a
        # landing that adds nothing reports nothing to publish.
        assert ("workspaces",) not in view.stale_calls()
        assert not view.install(view._generation, {("workspaces",): shown})
    finally:
        await _settle(screen)
