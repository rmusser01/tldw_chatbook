"""The Inspector's Library-activity projection reads durable rows off the UI loop.

TASK-33620.15.4 (measured 2026-10-23, base 5b5920b8e2): a Console send window
moved the projection's token (active path, selection, activity revision)
about three times, and each move read the conversation's whole trajectory
sidecar (``get_trajectory_rows``, storage admission included) synchronously
inside the control-bar projection, on the UI loop: 2.1 ms a send window,
3.4 ms for the read after the durable commit.

The durable rows are now read by a worker when their source (database,
conversation) or the store's activity revision moves, and the projection
renders from the rows that landed. These tests drive the real controller and
store over a file-backed ChaChaNotes database (an in-memory one keeps the
synchronous read: its connection is per thread) and pin:

* no ``get_trajectory_rows`` call on the UI loop's thread, one on a worker;
* freshness: the rail shows the durable rows as soon as the read lands, and
  a revision move never renders rows the store has already moved past;
* ordering: a read that lands after a newer one was requested publishes
  nothing;
* teardown: a read that lands after the screen closed publishes nothing;
* storage: a local pause neither renders nor caches, and the next projection
  reads again; any other failure degrades exactly as the synchronous read
  did (no durable rows, the same warning).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.library_activity import LibraryActivityEvent
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.library_activity import (
    ConsoleLibraryActivityController,
)

pytestmark = pytest.mark.bootstrap_profile


def _event(index: int) -> LibraryActivityEvent:
    return LibraryActivityEvent(
        version=1,
        event_id=f"event-{index}",
        attempt_id="attempt-1",
        run_id="run-1",
        actor_kind="primary",
        parent_run_id=None,
        library_provider="direct",
        operation="library_search_notes",
        status="succeeded",
        result_count=1,
        query_preview=f"query {index}",
        source_refs=(),
        error_code=None,
        error_summary=None,
    )


class _Rail:
    def __init__(self) -> None:
        self.views: list[tuple[str, ...]] = []

    def sync_library_activity(self, view, **_kwargs) -> None:
        self.views.append(tuple(item.event.event_id for item in view.actions))


class _Transcript:
    def __init__(self, selected: str) -> None:
        self.selected_message_id = selected
        self.counts: list[dict[str, int]] = []

    def thinking_owner_message_id(self, _message_id):
        return None

    def set_library_activity_counts(self, counts) -> None:
        self.counts.append(dict(counts))


class _Harness:
    """A durable conversation with one saved activity event, and the controller."""

    def __init__(self, tmp_path) -> None:
        self.db = CharactersRAGDB(tmp_path / "activity.sqlite", "activity-off-loop")
        self.store = ConsoleChatStore(persistence=ChatPersistenceService(self.db))
        session = self.store.create_session(ephemeral=True)
        self.session_id = session.id
        self.user = self.store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="question"
        )
        self.assistant = self.store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        self.conversation_id = self.store.promote_ephemeral_session(session.id)
        self.store.active_session_id = session.id
        self.store.admit_library_activity(session.id, self.user.id, _event(1))
        assert self.store.flush_library_activity(session.id).status == "saved"
        self.rail = _Rail()
        self.transcript = _Transcript(self.assistant.id)
        self.closed = False
        self.paused = False
        self.tasks: list[asyncio.Task] = []
        self.controller = ConsoleLibraryActivityController(
            app_instance=object(),
            ensure_store=lambda: self.store,
            transcript=lambda: self.transcript,
            inspector_rail=lambda: self.rail,
            citation_counts=lambda: {},
            reveal_inspector=lambda: None,
            sync_native_ui=self._never,
            notify=lambda *_args, **_kwargs: None,
        )
        # Set as attributes so the base build (no such parameters) constructs
        # the same controller and fails on the read's thread, not on a kwarg.
        self.controller.run_worker = self._schedule
        self.controller.presentation_paused = lambda: self.paused
        self.controller.torn_down = lambda: self.closed

    async def _never(self) -> None:
        raise AssertionError("the projection asked for a whole sync")

    def _schedule(self, work):
        task = asyncio.ensure_future(work)
        self.tasks.append(task)
        return task

    async def settle(self) -> None:
        while self.tasks:
            pending, self.tasks = self.tasks, []
            await asyncio.gather(*pending)

    def close(self) -> None:
        self.db.close()


class _ReadThreads:
    """Which threads ran ``get_trajectory_rows`` (observed, never replaced)."""

    def __init__(self) -> None:
        self.threads: list[int] = []
        self._code = CharactersRAGDB.get_trajectory_rows.__code__

    def __enter__(self):
        import sys

        self._previous = sys.getprofile(), threading.getprofile()

        def observe(frame, event, _argument):
            if event == "call" and frame.f_code is self._code:
                self.threads.append(threading.get_ident())

        threading.setprofile_all_threads(observe)
        return self

    def __exit__(self, *_exc):
        import sys

        threading.setprofile_all_threads(self._previous[1])
        sys.setprofile(self._previous[0])

    def on_main(self) -> int:
        main = threading.main_thread().ident
        return sum(1 for thread in self.threads if thread == main)


@pytest.mark.asyncio
async def test_projection_reads_durable_rows_on_a_worker_and_shows_them_when_they_land(
    tmp_path,
):
    harness = _Harness(tmp_path)
    try:
        with _ReadThreads() as reads:
            harness.controller.sync_projection()
            assert reads.on_main() == 0, (
                "the projection read the trajectory sidecar on the UI loop"
            )
            await harness.settle()
        assert len(reads.threads) == 1, reads.threads
        assert reads.on_main() == 0
        # Freshness: the landing renders the durable event at once.
        assert harness.rail.views[-1] == ("event-1",)
        assert harness.transcript.counts[-1] == {harness.assistant.id: 1}
        assert harness.controller.view.actions[0].event.event_id == "event-1"

        # A selection or active-path move with no revision move renders from
        # the landed rows at once, with no read at all.
        rendered = len(harness.rail.views)
        with _ReadThreads() as reads:
            harness.transcript.selected_message_id = harness.user.id
            harness.controller.sync_projection()
            await harness.settle()
        assert reads.threads == []
        assert len(harness.rail.views) == rendered + 1
        assert harness.rail.views[-1] == ("event-1",)
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_a_revision_move_renders_only_once_the_newer_rows_land(tmp_path):
    harness = _Harness(tmp_path)
    try:
        harness.controller.sync_projection()
        await harness.settle()
        rendered = len(harness.rail.views)
        harness.store.admit_library_activity(
            harness.session_id, harness.user.id, _event(2)
        )
        assert harness.store.flush_library_activity(harness.session_id).status == (
            "saved"
        )
        with _ReadThreads() as reads:
            harness.controller.sync_projection()
            # The flush moved the event out of the pending buffer; rows read
            # before it would drop it from view, so nothing renders yet.
            assert len(harness.rail.views) == rendered
            await harness.settle()
        assert reads.on_main() == 0 and len(reads.threads) == 1
        assert harness.rail.views[-1] == ("event-1", "event-2")
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_a_read_overtaken_by_a_newer_one_never_publishes(tmp_path):
    harness = _Harness(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = harness.db.get_trajectory_rows
    calls = []

    def held(conversation_id):
        calls.append(conversation_id)
        if len(calls) == 1 and threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(5)
        return original(conversation_id)

    harness.db.get_trajectory_rows = held
    try:
        harness.controller.sync_projection()
        assert harness.tasks, "the projection started no worker read"
        assert await asyncio.to_thread(entered.wait, 5)
        harness.store.admit_library_activity(
            harness.session_id, harness.user.id, _event(2)
        )
        harness.store.flush_library_activity(harness.session_id)
        harness.controller.sync_projection()  # A newer read, for the newer revision.
        newer = [task for task in harness.tasks if not task.done()]
        await asyncio.wait_for(asyncio.shield(newer[-1]), 5)
        assert harness.rail.views[-1] == ("event-1", "event-2")
        shown = list(harness.rail.views)
        release.set()
        await harness.settle()
        assert harness.rail.views == shown, "the older read overwrote the newer one"
        assert harness.controller.view.actions[-1].event.event_id == "event-2"
    finally:
        release.set()
        harness.close()


@pytest.mark.asyncio
async def test_a_read_landing_after_teardown_publishes_nothing(tmp_path):
    harness = _Harness(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = harness.db.get_trajectory_rows

    def held(conversation_id):
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(5)
        return original(conversation_id)

    harness.db.get_trajectory_rows = held
    try:
        harness.controller.sync_projection()
        assert harness.tasks, "the projection started no worker read"
        assert await asyncio.to_thread(entered.wait, 5)
        shown = list(harness.rail.views)
        harness.closed = True
        release.set()
        await harness.settle()
        assert harness.rail.views == shown
        assert harness.transcript.counts == []
    finally:
        release.set()
        harness.close()


@pytest.mark.asyncio
async def test_a_local_storage_pause_neither_renders_nor_caches_the_refused_read(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    harness = _Harness(tmp_path)
    original = harness.db.get_trajectory_rows
    refuse = {"on": True}

    def paused_read(conversation_id):
        if refuse["on"]:
            raise RecoveryRequired("storage_locally_paused")
        return original(conversation_id)

    harness.db.get_trajectory_rows = paused_read
    try:
        harness.controller.sync_projection()
        assert harness.tasks, "the projection started no worker read"
        await harness.settle()
        assert ("event-1",) not in harness.rail.views
        assert harness.transcript.counts == []
        refuse["on"] = False
        # The next projection (the pause's replayed sync) reads again.
        harness.controller.sync_projection()
        await harness.settle()
        assert harness.rail.views[-1] == ("event-1",)
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_any_other_read_failure_degrades_as_the_synchronous_read_did(
    tmp_path,
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError

    harness = _Harness(tmp_path)

    def broken(_conversation_id):
        raise CharactersRAGDBError("private detail")

    harness.db.get_trajectory_rows = broken
    try:
        harness.controller.sync_projection()
        assert harness.tasks, "the projection started no worker read"
        await harness.settle()
        # No durable rows: the saved event is not shown, as on the base.
        assert harness.rail.views[-1] == ()
        assert harness.controller.view.actions == ()
    finally:
        harness.close()
