"""MobileChatService — parent-process Console engine binding (ADR-234).

Task 3 of the served-mobile Phase 1b plan. Real engine over tmp_path SQLite
DBs; the provider is the Task-2 ``AgentModeGateway`` double (agent-capable:
emits one native ``fs_list`` tool call when tools are advertised, then the
final answer once the tool result returns). The service test IS the spike,
productized: the four tests below restate the Binding-1 probe's assertions
against the ``MobileChatService`` surface the HTTP/WS layer will consume.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from Tests.console_provider_doubles import AgentModeGateway

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Web_Server.mobile_service import MobileChatService, MobileSendRefused

# The real engine's submit path reads the guarded config loader (turn
# configuration, [tools] gates, hooks admission). Under pytest's per-test
# config redirect those reads fail closed with RecoveryRequired — the same
# admission signature Tests/conftest.py documents for the console suites
# (TASK-19425/22302) — so this module keeps the bootstrap config profile,
# like the real-engine Tests/Chat suites (e.g. test_automatic_wake_
# recovery.py). All databases stay tmp_path-owned.
pytestmark = pytest.mark.bootstrap_profile

FINAL_ANSWER = "MOBILE_AGENT_ROUND_TRIP_COMPLETE"
WORKSPACE_ID = "ws-mobile-1b"


async def _subscribe(service: MobileChatService, sink: list[Any]) -> None:
    """Background subscriber: append every fanned-out event to ``sink``."""
    async for event in service.events():
        sink.append(event)


async def _wait_for(predicate, timeout: float, what: str) -> None:
    """Poll ``predicate`` at 0.1 s until true; fail with ``what`` on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _bound_workspace(service: MobileChatService, tmp_path: Path) -> str:
    """One workspace + one folder binding through the service's registry.

    Mirrors the spike probe's ``ws-mobile-spike`` fixture: a workspace-bound
    chat is the realistic mobile case (default-workspace sessions get no
    ``fs_*``/``git_*`` tools at all).
    """
    folder = tmp_path / "code"
    (folder / "src").mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text("# mobile\n", encoding="utf-8")
    registry = service.workspace_registry
    registry.create_workspace(workspace_id=WORKSPACE_ID, name="Mobile 1b")
    registry.add_folder_binding(WORKSPACE_ID, folder, allow_write=True)
    return WORKSPACE_ID


async def _shutdown(service: MobileChatService, subscriber: asyncio.Task) -> None:
    subscriber.cancel()
    try:
        await subscriber
    except asyncio.CancelledError:
        pass
    await asyncio.wait_for(service.shutdown(), timeout=15)


async def test_agent_round_trip_with_approval(tmp_path: Path) -> None:
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    # Two subscribers: the first holds its own queue (fan-out sanity), the
    # second collects the events the asserts read.
    subscriber = asyncio.create_task(_subscribe(service, []))
    got: list[Any] = []
    collector = asyncio.create_task(_subscribe(service, got))
    try:
        workspace_id = _bound_workspace(service, tmp_path)
        session = await service.create_session(
            title="phone chat", workspace_id=workspace_id
        )
        assert session["workspace_id"] == workspace_id
        assert session["has_folder_tools"] is True

        send_task = asyncio.create_task(
            service.send(session["id"], "list the folder")
        )
        await _wait_for(
            lambda: any(e.type == "approval_requested" for e in got),
            timeout=30,
            what="approval_requested event",
        )
        # Non-blocking send (live-check round 1): the send call must have
        # returned already — the turn is parked on this approval, which
        # only this test can resolve, so a blocking send would still hang.
        assert send_task.done() and not send_task.exception()
        approvals = [e for e in got if e.type == "approval_requested"]
        payload = approvals[0].payload
        assert payload["calls"][0]["llm_name"] == "fs_list"
        assert payload["round_id"]

        # The pending round is also visible on the polling surface.
        listed = service.list_approvals()
        assert any(a["round_id"] == payload["round_id"] for a in listed)

        call = payload["calls"][0]
        assert (
            service.decide_approval(
                payload["round_id"], {call["call_id"]: "approve_once"}
            )
            is True
        )
        # Deciding an unknown round is reported, not raised.
        assert service.decide_approval("no-such-round", {"x": "deny"}) is False

        await asyncio.wait_for(send_task, timeout=60)
        await _wait_for(
            lambda: any(
                e.type == "run_completed" and e.session_id == session["id"]
                for e in got
            ),
            timeout=15,
            what="run_completed event",
        )

        messages = service.session_messages(session["id"])
        assert any(
            m["role"] == "assistant" and FINAL_ANSWER in m["content"]
            for m in messages["messages"]
        )
        # The tool really ran: the provider saw the fs_list result.
        assert gateway.tool_result_messages, "fs_list result never reached provider"

        deltas = [e for e in got if e.type == "delta" and e.session_id == session["id"]]
        assert deltas, "no streaming deltas observed during the run"

        # Wire shape: monotonic per-service seq, JSON-serializable payloads.
        seqs = [e.seq for e in got]
        assert seqs == sorted(seqs)
        for event in got:
            json.dumps(event.to_wire())
    finally:
        await _shutdown(service, collector)
        await _shutdown(service, subscriber)


async def test_send_is_serialized_per_session(tmp_path: Path) -> None:
    """Turns serialize per session; a concurrent send is refused (round 1).

    The non-blocking contract: ``send`` schedules the turn and returns, so
    a second send while the first turn is pending raises ``MobileSendRefused``
    (the engine's own one-run-per-session admission rule, applied at the
    service boundary instead of queueing on the lock). Once the first turn
    completes, the next send is accepted and the transcript order is
    preserved: each user row lands only after the previous turn's
    assistant row.
    """
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="serial")
        await service.send(session["id"], "first question")
        with pytest.raises(MobileSendRefused):
            await service.send(session["id"], "second question too early")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="first run_completed event",
        )

        await service.send(session["id"], "second question")
        await _wait_for(
            lambda: len([e for e in got if e.type == "run_completed"]) == 2,
            timeout=15,
            what="two run_completed events",
        )

        messages = service.session_messages(session["id"])["messages"]
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        assert user_texts == ["first question", "second question"]
        # The second user row landed only after the first turn's assistant
        # row — the turns really did serialize.
        roles = [m["role"] for m in messages]
        assert roles.count("user") == 2
        second_user_at = roles.index("user", 1)
        assert any(r == "assistant" for r in roles[:second_user_at])
        # Two full provider round-trips (one per turn; each turn's first
        # call attempts the gateway's fs_list call and fails fast because
        # the default workspace never advertises it, then the follow-up
        # call answers).
        assert len(gateway.stream_calls) >= 2
    finally:
        await _shutdown(service, subscriber)


async def test_send_returns_while_the_turn_is_pending(tmp_path: Path) -> None:
    """The 202-shape contract: send schedules and returns (finding 1).

    The gateway's first provider call parks on a gate only this test opens,
    so a send that blocked on ``submit_draft`` (the old shape — the one the
    live check caught) would hang until the timeout. Also proves the
    unknown-session pre-validation still raises before anything is
    scheduled, and that deltas + terminal events flow once the gate opens.
    """

    class _GatedAgentGateway(AgentModeGateway):
        """AgentModeGateway whose provider calls park on ``gate``."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            await self.gate.wait()
            async for chunk in super().stream_chat(resolution, messages, **kwargs):
                yield chunk

    gateway = _GatedAgentGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="pending turn")
        with pytest.raises(KeyError):
            await service.send("no-such-session", "gone")

        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.wait_for(
            service.send(session["id"], "held turn"), timeout=5
        )
        assert loop.time() - started < 2.0, "send blocked on the turn"

        with pytest.raises(MobileSendRefused):
            await service.send(session["id"], "too soon")

        gateway.gate.set()
        await _wait_for(
            lambda: any(
                e.type == "run_completed" and e.session_id == session["id"]
                for e in got
            ),
            timeout=30,
            what="run_completed after gate release",
        )
        deltas = [
            e for e in got if e.type == "delta" and e.session_id == session["id"]
        ]
        assert deltas, "no streaming deltas observed during the run"
        messages = service.session_messages(session["id"])["messages"]
        assert any(
            m["role"] == "user" and m["content"] == "held turn" for m in messages
        )
        assert any(
            m["role"] == "assistant" and FINAL_ANSWER in m["content"]
            for m in messages
        )
    finally:
        gateway.gate.set()  # never leave the parked turn holding shutdown
        await _shutdown(service, subscriber)


async def test_default_workspace_session_has_no_folder_tools(tmp_path: Path) -> None:
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="no folder")
        listing = service.sessions
        assert listing[0]["workspace_id"] == "global"
        assert listing[0]["has_folder_tools"] is False

        await service.send(session["id"], "hello without tools")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="run_completed event",
        )
        # Real evidence: builtin tools may still be advertised, but the
        # bound-folder path-tool family (fs_list) never is in the default
        # workspace.
        assert gateway.stream_calls, "provider was never called"
        assert all(not c["tool_advertised"] for c in gateway.stream_calls)
    finally:
        await _shutdown(service, subscriber)


async def test_messages_cursor(tmp_path: Path) -> None:
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="cursor")

        await service.send(session["id"], "turn one")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="first run_completed event",
        )
        first = service.session_messages(session["id"])
        first_ids = [m["id"] for m in first["messages"]]
        assert first["cursor"] == first_ids[-1]

        got.clear()
        await service.send(session["id"], "turn two")
        await _wait_for(
            lambda: len([e for e in got if e.type == "run_completed"]) == 1,
            timeout=15,
            what="second run_completed event",
        )

        after = service.session_messages(
            session["id"], after_message_id=first["cursor"]
        )
        after_ids = [m["id"] for m in after["messages"]]
        assert after_ids, "cursor returned no later messages"
        assert not (set(after_ids) & set(first_ids))
        assert after["cursor"] == after_ids[-1]
        # The second turn's user row is the first row past the cursor.
        assert after["messages"][0]["role"] == "user"
        assert after["messages"][0]["content"] == "turn two"
        assert any(
            m["role"] == "assistant" and FINAL_ANSWER in m["content"]
            for m in after["messages"]
        )
    finally:
        await _shutdown(service, subscriber)


async def test_send_after_run_completed_is_accepted_immediately(
    tmp_path: Path,
) -> None:
    """A send reacting to ``run_completed`` in its first slice is admitted.

    Deterministic pin for the admission half of the turn-slot race
    (re-review round 1). Mechanism, by construction: the event fan-out is
    synchronous (``put_nowait``), so the instant ``run_completed`` is
    emitted, this subscriber's ``Queue.get`` wakeup is queued on the loop's
    ready list. Pre-fix, the emit happened inside ``_run_turn`` while the
    session lock was held and the session was still in ``turns_in_flight``
    (cleanup lived only in the task-done callback) — and a finishing task's
    done-callback is queued strictly AFTER anything queued during the
    task's body. The subscriber therefore always ran first and its send
    deterministically raised ``MobileSendRefused``. The fix releases the
    admission slot BEFORE announcing terminal (no ``await`` between), so
    this send is always admitted.
    """
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    got: list[Any] = []
    outcome: list[str] = []
    session_id_holder: list[str] = []

    async def _reactive_subscriber() -> None:
        """On the first ``run_completed`` for the target session, send again
        immediately — in this wakeup slice, before anything else runs."""
        async for event in service.events():
            if (
                event.type == "run_completed"
                and event.session_id in session_id_holder
            ):
                assert outcome == [], "second run_completed before send settled"
                try:
                    await service.send(event.session_id, "immediate follow-up")
                except MobileSendRefused:
                    outcome.append("refused")
                else:
                    outcome.append("accepted")
                return

    subscriber = asyncio.create_task(_subscribe(service, got))
    reactive = asyncio.create_task(_reactive_subscriber())
    try:
        session = await service.create_session(title="reactive send")
        session_id_holder.append(session["id"])
        await service.send(session["id"], "first turn")
        await _wait_for(
            lambda: len(outcome) == 1,
            timeout=15,
            what="reactive send outcome after first run_completed",
        )
        # The pin: the immediate send was admitted, never refused.
        assert outcome == ["accepted"], outcome
        # And its turn really completed (second terminal event observed).
        await _wait_for(
            lambda: len([e for e in got if e.type == "run_completed"]) == 2,
            timeout=15,
            what="second run_completed event",
        )
        messages = service.session_messages(session["id"])["messages"]
        assert [m["content"] for m in messages if m["role"] == "user"] == [
            "first turn",
            "immediate follow-up",
        ]
    finally:
        await _shutdown(service, reactive)
        await _shutdown(service, subscriber)


async def test_pump_tick_in_the_return_window_delivers_fast_run_content(
    tmp_path: Path,
) -> None:
    """A pump tick inside the return window delivers the content; it must
    not baseline it away.

    Deterministic pin for the streaming half of the turn-slot race
    (re-review round 1). Mechanism: in agent mode the assistant row only
    becomes scannable with content at finalize, so the first pump tick
    that can ever see the message sees it already terminal, while the
    owning turn still holds its admission slot inside ``submit_draft``'s
    finalize-to-flush window. Two facts make a naive timing pin useless
    (both observed live): idle, that window is sub-millisecond — a tick
    only lands in it under load, which is how the original flake surfaced
    at the production 0.5 s cadence; and a tick that lands MID-run
    instead creates a pending track, which lets even pre-fix code deliver
    via the tracked path. So this test removes both timing inputs: the
    real pump is cancelled (the tick below is the only scan) and the
    service's own flush seam is wrapped so that when the turn performs
    its send-time flush, a real ``_poll_once`` tick runs FIRST — flush
    time is inside the window by construction, with ``track`` guaranteed
    None. Pre-fix, that tick scanned the session as inactive, took the
    silent first-observation baseline, and the flush then had zero
    remaining content: no ``delta`` and no ``message_done`` ever fired
    (and the tick announced ``run_completed`` before any content). The
    fix treats the slot-owned session as live: full delta +
    ``message_done`` first, terminal announcement after.
    """
    gateway = AgentModeGateway()
    service = MobileChatService(data_dir=tmp_path, provider_gateway=gateway)
    # Build the engine, then kill the real pump: the wrapper's tick below
    # becomes the only scan, so the pre-flush track state is deterministic.
    await service._ensure_started()
    assert service._pump_task is not None
    service._pump_task.cancel()
    try:
        await service._pump_task
    except asyncio.CancelledError:
        pass
    original_scan = service._scan_stream

    def _tick_then_scan(engine, session_id, *, active, flush=False):
        if flush:
            # A pump tick lands in the return window just before the turn's
            # own flush would deliver — the exact interleaving the race
            # window produces when a tick gets scheduled inside it.
            service._poll_once(engine)
        return original_scan(engine, session_id, active=active, flush=flush)

    service._scan_stream = _tick_then_scan

    got: list[Any] = []
    subscriber = asyncio.create_task(_subscribe(service, got))
    try:
        session = await service.create_session(title="fast run")
        await service.send(session["id"], "answer fast")
        await _wait_for(
            lambda: any(e.type == "run_completed" for e in got),
            timeout=15,
            what="run_completed event",
        )

        deltas = [
            e
            for e in got
            if e.type == "delta" and e.session_id == session["id"]
        ]
        dones = [
            e
            for e in got
            if e.type == "message_done" and e.session_id == session["id"]
        ]
        assert deltas, "pump tick in the return window baselined the content"
        assert dones, "terminal message_done never fired for the fast run"
        delivered = "".join(e.payload["text"] for e in deltas)
        assert delivered == FINAL_ANSWER, delivered
        # The deltas and the terminal message_done reference one message.
        assert {e.payload["message_id"] for e in deltas} == {
            e.payload["message_id"] for e in dones
        }
        # Content precedes the terminal announcement — the tick cannot
        # announce run_completed for content the subscriber never got.
        types = [e.type for e in got]
        assert types.index("delta") < types.index("run_completed"), types
    finally:
        await _shutdown(service, subscriber)


# ========================================================================= #
# Reading surface (Phase 2b): notes + library reads over tmp_path DBs       #
# ========================================================================= #
#
# The four tests below seed real ``CharactersRAGDB`` / ``MediaDatabase``
# files under tmp_path and hand their paths to the service through the
# ``notes_db_path`` / ``media_db_path`` injection seams — the service's
# default resolution is the app's own config helpers, which these tests
# deliberately never touch (plan constraint: tests avoid config-loading
# paths). No engine is built: the reading methods stand alone.

_NOTES_DB_NAME = "reading-notes.sqlite"
_MEDIA_DB_NAME = "reading-media.sqlite"
_TRANSCRIPT_TEXT = "MOBILE_READING_TRANSCRIPT_BODY"


def _note_body(index: int) -> str:
    """One distinct note body comfortably past the 200-char preview cut."""
    return f"Reading note {index} body. " + (
        f"filler sentence {index} about the mobile reading surface. " * 8
    )


def _seed_reading_dbs(
    tmp_path: Path,
) -> tuple[
    CharactersRAGDB,
    list[str],
    dict[str, str],
    MediaDatabase,
    list[int],
]:
    """Seed the reading DBs at the exact paths the service will be given.

    3 notes with distinct bodies (stamped 10 ms apart so the
    ``last_modified DESC`` page order is deterministic at the DBs'
    millisecond timestamp precision) and 2 media rows, the first with one
    transcript. Notes go through ``add_note``; media through
    ``add_media_with_keywords``. The transcript lands via a direct INSERT —
    ``MediaDatabase`` has no public add-transcript API (ingestion owns that
    table); this is the established seeding idiom
    (``Tests/MCP/test_tools_resources_prompts_real_methods.py``).
    """
    notes_db = CharactersRAGDB(
        str(tmp_path / _NOTES_DB_NAME), "mobile-reading-fixture"
    )
    note_ids: list[str] = []
    bodies: dict[str, str] = {}
    for index in range(3):
        note_id = notes_db.add_note(
            title=f"Reading note {index}", content=_note_body(index)
        )
        assert note_id is not None
        note_ids.append(note_id)
        bodies[note_id] = _note_body(index)
        time.sleep(0.01)

    media_db = MediaDatabase(
        str(tmp_path / _MEDIA_DB_NAME), "mobile-reading-fixture"
    )
    media_ids: list[int] = []
    for index in range(2):
        media_id, _uuid, _message = media_db.add_media_with_keywords(
            title=f"Reading media {index}",
            content=f"reading media body {index}",
            media_type="article",
            keywords=[],
        )
        assert media_id is not None
        media_ids.append(int(media_id))
        time.sleep(0.01)
    media_db.execute_query(
        """
        INSERT INTO Transcripts
            (media_id, whisper_model, transcription, created_at, uuid,
             last_modified, version, client_id, deleted)
        VALUES (?, 'test-model', ?, datetime('now'), ?, datetime('now'), 1,
                'mobile-reading-fixture', 0)
        """,
        (media_ids[0], _TRANSCRIPT_TEXT, f"transcript-uuid-{media_ids[0]}"),
        commit=True,
    )
    return notes_db, note_ids, bodies, media_db, media_ids


def _reading_service(tmp_path: Path) -> MobileChatService:
    """A service whose reading DBs are the seeded tmp_path files."""
    return MobileChatService(
        data_dir=tmp_path / "engine",
        provider_gateway=AgentModeGateway(),
        notes_db_path=tmp_path / _NOTES_DB_NAME,
        media_db_path=tmp_path / _MEDIA_DB_NAME,
    )


async def test_notes_page_paginates_with_truncated_previews(
    tmp_path: Path,
) -> None:
    """notes_page pages newest-first with 200-char previews + has_more."""
    notes_db, note_ids, bodies, media_db, _ = _seed_reading_dbs(tmp_path)
    service = _reading_service(tmp_path)
    try:
        page1 = service.notes_page(page=1, per_page=2)
        assert page1["page"] == 1
        assert len(page1["items"]) == 2
        assert page1["has_more"] is True
        # Newest (last-added) note leads page 1: list_notes orders by
        # last_modified DESC and the seed spaces the stamps apart.
        assert page1["items"][0]["id"] == note_ids[-1]

        page2 = service.notes_page(page=2, per_page=2)
        assert page2["page"] == 2
        assert len(page2["items"]) == 1
        assert page2["has_more"] is False

        # Every note appears exactly once across the two pages.
        items = page1["items"] + page2["items"]
        assert sorted(item["id"] for item in items) == sorted(note_ids)
        for item in items:
            body = bodies[item["id"]]
            assert item["title"]
            assert item["preview"] != body, "preview must not be the full body"
            assert body.startswith(item["preview"])
            assert len(item["preview"]) <= 200
            # Timestamps pass through as-is.
            assert item["last_modified"]
            assert item["created_at"]
        json.dumps(page1)  # the whole page is JSON-safe
    finally:
        await service.shutdown()
        notes_db.close_connection()
        media_db.close_connection()


async def test_note_detail_returns_full_body_and_none_for_unknown(
    tmp_path: Path,
) -> None:
    """note_detail hands back the full row; unknown ids return None."""
    notes_db, note_ids, bodies, media_db, _ = _seed_reading_dbs(tmp_path)
    service = _reading_service(tmp_path)
    try:
        detail = service.note_detail(note_ids[1])
        assert detail is not None
        assert detail["id"] == note_ids[1]
        assert detail["title"] == "Reading note 1"
        assert detail["content"] == bodies[note_ids[1]]
        assert detail["last_modified"]
        json.dumps(detail)
        assert service.note_detail("no-such-note-id") is None
    finally:
        await service.shutdown()
        notes_db.close_connection()
        media_db.close_connection()


async def test_library_page_lists_media_with_pagination(
    tmp_path: Path,
) -> None:
    """library_page lists active media newest-first with total-based has_more."""
    notes_db, _, _, media_db, media_ids = _seed_reading_dbs(tmp_path)
    service = _reading_service(tmp_path)
    try:
        page1 = service.library_page(page=1, per_page=1)
        assert page1["page"] == 1
        assert [item["media_id"] for item in page1["items"]] == [media_ids[1]]
        assert page1["items"][0]["title"] == "Reading media 1"
        assert page1["items"][0]["media_type"] == "article"
        assert page1["items"][0]["last_modified"]
        assert page1["has_more"] is True  # total is 2

        page2 = service.library_page(page=2, per_page=1)
        assert [item["media_id"] for item in page2["items"]] == [media_ids[0]]
        assert page2["has_more"] is False
        json.dumps(page1)
    finally:
        await service.shutdown()
        notes_db.close_connection()
        media_db.close_connection()


async def test_library_detail_returns_row_with_transcript_and_none_for_unknown(
    tmp_path: Path,
) -> None:
    """library_detail joins the media row with its transcripts; None when absent."""
    notes_db, _, _, media_db, media_ids = _seed_reading_dbs(tmp_path)
    service = _reading_service(tmp_path)
    try:
        detail = service.library_detail(media_ids[0])
        assert detail is not None
        assert detail["media_id"] == media_ids[0]
        assert detail["title"] == "Reading media 0"
        assert detail["media_type"] == "article"
        assert any(
            _TRANSCRIPT_TEXT in t["transcription"] for t in detail["transcripts"]
        )
        json.dumps(detail)
        assert service.library_detail(987654321) is None
    finally:
        await service.shutdown()
        notes_db.close_connection()
        media_db.close_connection()
