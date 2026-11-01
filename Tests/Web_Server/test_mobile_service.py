"""Canonical Library reading through noncreating native SQLite owners."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError as MediaDatabaseError
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Web_Server.mobile_service import CanonicalMobileReads

pytestmark = pytest.mark.bootstrap_profile

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
    notes_db = CharactersRAGDB(str(tmp_path / _NOTES_DB_NAME), "mobile-reading-fixture")
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

    media_db = MediaDatabase(str(tmp_path / _MEDIA_DB_NAME), "mobile-reading-fixture")
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


def _reading_service(tmp_path: Path):
    return CanonicalMobileReads(
        tmp_path / _NOTES_DB_NAME, tmp_path / _MEDIA_DB_NAME, tmp_path / "agent_runs.db"
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
        service.close()
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
        service.close()
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
        service.close()
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
        service.close()
        notes_db.close_connection()
        media_db.close_connection()


@pytest.mark.parametrize("owner", [CharactersRAGDB, MediaDatabase])
def test_read_only_missing_store_is_not_created(tmp_path, owner):
    missing = tmp_path / "missing.db"
    with pytest.raises((CharactersRAGDBError, MediaDatabaseError)):
        owner.open_read_only(missing)
    assert not missing.exists()


@pytest.mark.parametrize("owner", [CharactersRAGDB, MediaDatabase])
def test_read_only_native_handle_cannot_mutate_or_migrate(tmp_path, owner, monkeypatch):
    path = tmp_path / "existing.db"
    writer = owner(path, client_id="seed")
    writer.close_connection()

    def migration_trap(self):
        raise AssertionError("Read must never initialize a schema")

    monkeypatch.setattr(owner, "_initialize_schema", migration_trap)
    reader = owner.open_read_only(path)
    try:
        connection = reader._get_thread_connection()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE must_not_exist (id INTEGER)")
    finally:
        reader.close_connection()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


@pytest.mark.parametrize("owner", [CharactersRAGDB, MediaDatabase])
def test_read_only_incompatible_schema_stays_unavailable(tmp_path, owner):
    path = tmp_path / "existing.db"
    writer = owner(path, client_id="seed")
    table = "db_schema_version" if owner is CharactersRAGDB else "schema_version"
    writer.execute_query(f"UPDATE {table} SET version=0", commit=True)
    writer.close_connection()
    before = path.read_bytes()
    with pytest.raises(Exception, match="incompatible"):
        owner.open_read_only(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("owner", [CharactersRAGDB, MediaDatabase])
def test_read_only_cached_handle_obeys_ordinary_pause(tmp_path, owner):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission

    path = tmp_path / "existing.db"
    writer = owner(path, client_id="seed")
    writer.close_connection()
    reader = owner.open_read_only(path)
    pause = storage_admission._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            reader._get_thread_connection()
    finally:
        pause.resume()
        reader.close_connection()


@pytest.mark.asyncio
async def test_nonstarting_discovery_replaces_retired_cache_and_fences_old_host():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_remote_commands import ConsoleRemoteCommands
    from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity
    from tldw_chatbook.Web_Server.mobile_service import MobileChatService
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer
    from tldw_chatbook.Web_Server.served_console_hosts import (
        ServedConsoleHost,
        ServedConsoleHosts,
    )

    starts = []

    class Service:
        def __init__(self, **callbacks):
            self.app_service_id = "native-child-" + str(len(starts))
            self._process = SimpleNamespace(returncode=None)

        async def start(self, width, height):
            starts.append(self.app_service_id)

        async def stop(self):
            self._process.returncode = 0

    hosts = ServedConsoleHosts(lambda identity: ServedConsoleHost(Service))
    buffer = ConsoleEventBuffer()

    class Broker:
        async def wait_connected(self, child_id, **kwargs):
            pass

        def console_events(self, child_id):
            return buffer

        async def request(self, child_id, method, payload, **kwargs):
            identity = ConsoleHostIdentity(
                server_epoch="server-epoch", console_id=child_id
            )
            if method == "console.snapshot.request":
                return SimpleNamespace(
                    payload={
                        "snapshot": dict(
                            identity.model_dump(),
                            sequence=0,
                            reset_required=False,
                            execution_admitted=True,
                        )
                    }
                )
            if method == "console.ticket.request":
                native = ConsoleRemoteCommands(
                    SimpleNamespace(
                        _raise_if_disposed_or_session_fenced=lambda _: None
                    ),
                    identity,
                )
                native._require_host(
                    ConsoleHostIdentity.model_validate(payload["request"]["target"])
                )
                raise AssertionError("Old-host ticket reached issuance")
            return SimpleNamespace(payload={})

    service = MobileChatService(
        hosts=hosts,
        storage_identity="canonical",
        broker=Broker(),
        readers=SimpleNamespace(close=lambda: None),
        require_cutover=lambda: None,
    )
    try:
        old = await hosts.get_or_start("canonical")
        old_identity = await service.bootstrap_host()
        await old.aclose()
        current = await hosts.get_or_start("canonical")
        assert current is not old and len(starts) == 2
        snapshot = await service.snapshot()
        assert snapshot["console_id"] == current.service.app_service_id
        stream = service.events(client_id="reconnecting-phone")
        await stream.aclose()
        assert len(starts) == 2
        with pytest.raises(ValueError, match="owner changed"):
            await service.issue_command_ticket(
                "create", old_identity, {"workspace_id": "global"}
            )
        assert len(starts) == 2
    finally:
        await service.shutdown()
        await hosts.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["registration", "cleanup"])
async def test_event_stream_pins_exact_child_across_successor_discovery(replacement):
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.Web_Server.mobile_service import MobileChatService
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

    hosts = SimpleNamespace(
        current=SimpleNamespace(
            _closed=False, service=SimpleNamespace(app_service_id="child-A")
        )
    )
    hosts.get_existing = lambda identity: hosts.current
    entered, release = asyncio.Event(), asyncio.Event()
    buffers = {child: ConsoleEventBuffer() for child in ("child-A", "child-B")}
    calls = []

    class Broker:
        async def wait_connected(self, child_id, **kwargs):
            entered.set()
            await release.wait()

        def console_events(self, child_id):
            return buffers[child_id]

        async def request(self, child_id, method, payload, **kwargs):
            calls.append((child_id, payload["request"]["operation"]))
            return SimpleNamespace(payload={})

    service = MobileChatService(
        hosts=hosts,
        storage_identity="canonical",
        broker=Broker(),
        readers=SimpleNamespace(close=lambda: None),
        require_cutover=lambda: None,
    )
    stream = service.events(client_id="same-phone")
    first = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if replacement == "registration":
            hosts.current = SimpleNamespace(
                _closed=False, service=SimpleNamespace(app_service_id="child-B")
            )
        buffers["child-A"].publish("console.changed", {"sequence": 1})
        release.set()
        event = await asyncio.wait_for(first, 1)
        assert event.type == "console.changed"
        hosts.current = SimpleNamespace(
            _closed=False, service=SimpleNamespace(app_service_id="child-B")
        )
    finally:
        release.set()
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await stream.aclose()
        await service.shutdown()
    assert calls == [("child-A", "register"), ("child-A", "disconnect")]
