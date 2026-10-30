"""The per-turn Automatic Library outcome line's projection (TASK-33621.51).

``ConsoleLibraryOutcomeLines`` projects the persisted ``library_preparation``
record of each persisted USER turn on the active path onto that turn's native
message id. These tests drive it over the real store and a real file-backed
ChaChaNotes database and pin:

* copy: each outcome's line, the bypass reason included;
* the record is the only source, so the line follows a branch switch and
  survives a restart (a new store, connection and projection);
* cost: a file-backed conversation is read on a worker, once per key, and a
  sync with the same turns reads nothing; an in-memory database and a send's
  start read on the loop instead;
* a landed read repaints; one landing after teardown or during a storage
  pause publishes nothing.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest

from Tests.UI.test_console_library_activity_off_loop import _ReadThreads
from Tests.UI.test_console_resume_active_path import _resume_into_store
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.library_preparation import (
    LIBRARY_PREPARATION_EVENT_KIND,
    LibraryPreparationView,
    encode_library_preparation_event,
    library_preparation_event_for_outcome,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite
from tldw_chatbook.UI.Console_Modules.library_activity import (
    ConsoleLibraryOutcomeLines,
    library_outcome_copy,
)

pytestmark = pytest.mark.bootstrap_profile

ZERO = "Library: 0 matches · sent without Library evidence"


@pytest.mark.parametrize(
    ("outcome", "count", "reason", "line"),
    [
        ("evidence_sent", 1, None, "Library: 1 source sent"),
        ("evidence_sent", 4, None, "Library: 4 sources sent"),
        ("zero_matches", 0, None, ZERO),
        (
            "bypassed",
            0,
            "timed_out",
            "Library: sent once without Library · search timed out",
        ),
        ("bypassed", 0, "stopped", "Library: sent once without Library · search stopped"),
        ("bypassed", 0, "failed", "Library: sent once without Library · search failed"),
        # A record written before reasons were kept still says what happened.
        ("bypassed", 0, None, "Library: sent once without Library"),
    ],
)
def test_each_outcome_has_its_line(outcome, count, reason, line) -> None:
    view = LibraryPreparationView(
        turn_id="turn-1",
        outcome=outcome,
        result_count=count,
        source_types=("notes",),
        bypass_reason=reason,
    )

    assert library_outcome_copy(view) == line


def _payload(outcome: str, *, count: int = 0, reason: str | None = None) -> str:
    return encode_library_preparation_event(
        library_preparation_event_for_outcome(
            outcome,
            attempt_id="attempt-1",
            result_count=count,
            source_types=("notes", "media", "conversations"),
            bypass_reason=reason,
        )
    )


class _Transcript:
    def __init__(self) -> None:
        self.pushed: list[dict[str, str]] = []

    def set_library_outcome_lines(self, lines) -> None:
        self.pushed.append(dict(lines))


class _Harness:
    """A saved two-turn chat; each USER turn may carry a preparation record."""

    def __init__(self, tmp_path, *, worker: bool = True) -> None:
        self.path = tmp_path / "outcome.sqlite"
        self.db = CharactersRAGDB(self.path, "outcome-lines")
        self.store = ConsoleChatStore(persistence=ChatPersistenceService(self.db))
        session = self.store.create_session(ephemeral=True)
        self.session_id = session.id
        for role, content in (
            (ConsoleMessageRole.USER, "first question"),
            (ConsoleMessageRole.ASSISTANT, "first answer"),
            (ConsoleMessageRole.USER, "second question"),
            (ConsoleMessageRole.ASSISTANT, "second answer"),
        ):
            self.store.append_message(session.id, role=role, content=content)
        self.conversation_id = self.store.promote_ephemeral_session(session.id)
        self.store.active_session_id = session.id
        self.messages = self.store.messages_for_session(session.id)
        self.users = [m for m in self.messages if m.role is ConsoleMessageRole.USER]
        assert all(m.persisted_message_id for m in self.users)
        self.tasks: list[asyncio.Task] = []
        self.repaints = 0
        self.closed = False
        self.paused = False
        self.live = False
        self.lines = ConsoleLibraryOutcomeLines(
            ensure_store=lambda: self.store,
            run_worker=self._schedule if worker else None,
            repaint=self._repaint,
            live_reads=lambda: self.live,
            presentation_paused=lambda: self.paused,
            torn_down=lambda: self.closed,
        )

    def record(self, user, payload: str, *, seq: int) -> None:
        self.db.upsert_trajectory_rows(
            [
                TrajectoryRowWrite(
                    message_id=user.persisted_message_id,
                    conversation_id=self.conversation_id,
                    turn_id=user.persisted_message_id,
                    seq=seq,
                    event_kind=LIBRARY_PREPARATION_EVENT_KIND,
                    payload_json=payload,
                )
            ]
        )

    def _schedule(self, work):
        task = asyncio.ensure_future(work)
        self.tasks.append(task)
        return task

    async def _repaint(self) -> None:
        self.repaints += 1

    async def settle(self) -> None:
        while self.tasks:
            pending, self.tasks = self.tasks, []
            await asyncio.gather(*pending)

    def close(self) -> None:
        self.db.close()


@pytest.mark.asyncio
async def test_lines_are_read_on_a_worker_once_and_repaint_when_they_land(tmp_path):
    harness = _Harness(tmp_path)
    first, second = harness.users
    harness.record(first, _payload("evidence_sent", count=2), seq=50)
    harness.record(second, _payload("bypassed", reason="timed_out"), seq=51)
    try:
        transcript = _Transcript()
        with _ReadThreads() as reads:
            assert harness.lines.sync_transcript(transcript, harness.messages) == {}
            assert reads.on_main() == 0, "the projection read on the UI loop"
            await harness.settle()
        assert len(reads.threads) == 1 and reads.on_main() == 0
        assert harness.repaints == 1

        expected = {
            first.id: "Library: 2 sources sent",
            second.id: "Library: sent once without Library · search timed out",
        }
        with _ReadThreads() as reads:
            for _ in range(3):  # ordinary syncs with the same turns
                assert harness.lines.sync_transcript(transcript, harness.messages) == (
                    expected
                )
            await harness.settle()
        assert reads.threads == []
        assert transcript.pushed[-1] == expected
        assert harness.repaints == 1
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_a_newly_persisted_turn_is_read_once_and_kept_lines_stay_meanwhile(
    tmp_path,
):
    harness = _Harness(tmp_path)
    first, second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    # The second turn is not accepted yet: no durable id, no record.
    pending = [
        replace(m, persisted_message_id=None) if m.id == second.id else m
        for m in harness.messages
    ]
    try:
        harness.lines.lines_for(pending)
        await harness.settle()
        assert harness.lines.lines_for(pending) == {first.id: ZERO}

        # Acceptance writes the record and gives the turn its durable id.
        harness.record(second, _payload("evidence_sent", count=1), seq=51)
        with _ReadThreads() as reads:
            # The first turn's line stays while the newer read is in flight.
            assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
            await harness.settle()
        assert len(reads.threads) == 1
        assert harness.lines.lines_for(harness.messages) == {
            first.id: ZERO,
            second.id: "Library: 1 source sent",
        }
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_the_line_follows_a_branch_switch(tmp_path):
    """Each branch shows only its own turns' records."""
    harness = _Harness(tmp_path)
    first, second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    harness.record(second, _payload("bypassed", reason="stopped"), seq=51)
    branch_a = [m for m in harness.messages if m.id != second.id]
    branch_b = [m for m in harness.messages if m.id != first.id]
    try:
        for branch, expected in (
            (branch_a, {first.id: ZERO}),
            (
                branch_b,
                {second.id: "Library: sent once without Library · search stopped"},
            ),
            (branch_a, {first.id: ZERO}),
        ):
            harness.lines.lines_for(branch)
            await harness.settle()
            assert harness.lines.lines_for(branch) == expected
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_the_line_survives_a_restart(tmp_path):
    """A new store, connection and projection read the same record."""
    harness = _Harness(tmp_path)
    first, _second = harness.users
    harness.record(first, _payload("evidence_sent", count=3), seq=50)
    harness.close()

    db = CharactersRAGDB(harness.path, "outcome-lines-restart")
    # The production resume plumbing: full tree, stored cursor, restored session.
    store, session = _resume_into_store(db, harness.conversation_id)
    store.active_session_id = session.id
    messages = store.messages_for_session(session.id)
    resumed_first = next(
        m for m in messages if m.persisted_message_id == first.persisted_message_id
    )
    restarted = ConsoleLibraryOutcomeLines(ensure_store=lambda: store)
    try:
        assert restarted.lines_for(messages) == {
            resumed_first.id: "Library: 3 sources sent"
        }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_inline_reads_for_a_send_start_and_an_in_memory_database(tmp_path):
    harness = _Harness(tmp_path)
    first, _second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    try:
        harness.live = True  # a send's start keeps display reads on the loop
        with _ReadThreads() as reads:
            assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
        assert reads.on_main() == 1 and harness.tasks == []
    finally:
        harness.close()

    memory = CharactersRAGDB(":memory:", "outcome-lines-memory")
    store = ConsoleChatStore(persistence=ChatPersistenceService(memory))
    session = store.create_session(ephemeral=True)
    user = store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a")
    conversation_id = store.promote_ephemeral_session(session.id)
    store.active_session_id = session.id
    messages = store.messages_for_session(session.id)
    durable = next(m for m in messages if m.id == user.id).persisted_message_id
    memory.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=durable,
                conversation_id=conversation_id,
                turn_id=durable,
                seq=50,
                event_kind=LIBRARY_PREPARATION_EVENT_KIND,
                payload_json=_payload("zero_matches"),
            )
        ]
    )
    scheduled = []
    lines = ConsoleLibraryOutcomeLines(
        ensure_store=lambda: store, run_worker=scheduled.append
    )
    assert lines.lines_for(messages) == {user.id: ZERO}
    assert scheduled == []  # its connection is per thread: never a worker


@pytest.mark.asyncio
async def test_a_read_landing_after_teardown_or_in_a_storage_pause_publishes_nothing(
    tmp_path,
):
    harness = _Harness(tmp_path)
    first, _second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    try:
        harness.lines.lines_for(harness.messages)
        harness.closed = True
        await harness.settle()
        assert harness.repaints == 0
        harness.closed = False

        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        original = harness.db.get_trajectory_rows
        reads = []

        def paused(conversation_id):
            reads.append(threading.get_ident())
            raise RecoveryRequired("storage_locally_paused")

        harness.db.get_trajectory_rows = paused
        harness.lines.lines_for(harness.messages)
        await harness.settle()
        assert harness.lines.lines_for(harness.messages) == {}
        assert harness.repaints == 0
        assert len(reads) == 1

        # The pause ends: the next sync reads again and shows the line.
        harness.db.get_trajectory_rows = original
        harness.lines.lines_for(harness.messages)
        await harness.settle()
        assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
        assert harness.repaints == 1
    finally:
        harness.close()


def _pending_second(harness) -> list:
    """The two-turn chat before its second USER turn was accepted."""
    second = harness.users[1]
    return [
        replace(m, persisted_message_id=None) if m.id == second.id else m
        for m in harness.messages
    ]


@pytest.mark.asyncio
async def test_a_failed_worker_read_keeps_the_lines_already_shown(tmp_path):
    """A transient read failure must not blank every turn's line.

    A record never changes once written, so the lines a previous read landed
    stay right; the failed key is marked read so the next syncs do not retry
    it on every tick, and nothing repaints because nothing changed.
    """
    import sqlite3

    harness = _Harness(tmp_path)
    first, second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    pending = _pending_second(harness)
    try:
        harness.lines.lines_for(pending)
        await harness.settle()
        assert harness.lines.lines_for(pending) == {first.id: ZERO}
        assert harness.repaints == 1

        harness.record(second, _payload("evidence_sent", count=1), seq=51)
        calls = []

        def busy(conversation_id):
            calls.append(conversation_id)
            raise sqlite3.OperationalError("database is locked")

        harness.db.get_trajectory_rows = busy
        assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
        await harness.settle()
        for _ in range(3):  # ordinary syncs after the failure
            assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
            await harness.settle()
        assert len(calls) == 1, "the failed key was retried on every sync"
        assert harness.repaints == 1
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_a_failed_inline_read_keeps_lines_and_a_paused_one_reads_again(
    tmp_path,
):
    """The loop-side read degrades like the worker's: kept lines, no blanking.

    Any failure keeps the landed lines. A local storage pause lands nothing,
    so the first sync after it ends reads again and shows the new turn's line;
    any other failure marks the key read so it is not retried every sync.
    """
    import sqlite3

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    harness = _Harness(tmp_path)
    first, second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    harness.live = True  # a send's start reads on the loop
    pending = _pending_second(harness)
    original = harness.db.get_trajectory_rows
    try:
        assert harness.lines.lines_for(pending) == {first.id: ZERO}
        harness.record(second, _payload("evidence_sent", count=2), seq=51)

        calls = []

        def paused(conversation_id):
            calls.append(conversation_id)
            raise RecoveryRequired("storage_locally_paused")

        harness.db.get_trajectory_rows = paused
        assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
        assert harness.lines.lines_for(harness.messages) == {first.id: ZERO}
        assert len(calls) == 2, "a storage pause must not mark the key read"

        harness.db.get_trajectory_rows = original
        both = {first.id: ZERO, second.id: "Library: 2 sources sent"}
        assert harness.lines.lines_for(harness.messages) == both

        # A different failure on a new key (the branch without the first
        # turn) keeps what is shown for turns still on the path and is not
        # retried on every sync.
        branch = [m for m in harness.messages if m.id != first.id]

        def busy(conversation_id):
            calls.append(conversation_id)
            raise sqlite3.OperationalError("database is locked")

        harness.db.get_trajectory_rows = busy
        calls.clear()
        for _ in range(3):
            assert harness.lines.lines_for(branch) == {
                second.id: "Library: 2 sources sent"
            }
        assert len(calls) == 1
        assert harness.tasks == []
    finally:
        harness.db.get_trajectory_rows = original
        harness.close()


@pytest.mark.asyncio
async def test_an_older_worker_read_never_overwrites_a_newer_inline_landing(
    tmp_path,
):
    """A send's start lands its newer key inline while an older read is out.

    The older worker read, landing afterwards, is stale: it must neither
    replace the newer lines nor force a repaint.
    """
    harness = _Harness(tmp_path)
    first, second = harness.users
    harness.record(first, _payload("zero_matches"), seq=50)
    pending = _pending_second(harness)
    both = {first.id: ZERO, second.id: "Library: 1 source sent"}
    try:
        harness.lines.lines_for(pending)  # older key: a worker read is out
        assert len(harness.tasks) == 1

        harness.record(second, _payload("evidence_sent", count=1), seq=51)
        harness.live = True
        assert harness.lines.lines_for(harness.messages) == both
        harness.live = False

        await harness.settle()  # the older read lands now
        assert harness.repaints == 0, "a stale read forced a repaint"
        with _ReadThreads() as reads:
            assert harness.lines.lines_for(harness.messages) == both
            await harness.settle()
        assert reads.threads == [], "a stale read regressed the landed key"
    finally:
        harness.close()
