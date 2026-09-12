"""TASK-23027: the narrow (version, deleted) projection behind note reuse.

Layer by layer: ``CharactersRAGDB.get_note_version_states`` (chunked, one
snapshot, tombstones included), the ``NotesInteropService`` wrapper, the
``NotesScopeService`` seam (bulk preferred, per-note fallback for backends
without the projection, scope gate), and ``NotesScopeSyncAuthority.
observe_versions`` (deleted rows excluded at its own level -- the runtime's
version comparison would also catch a tombstone because deletion bumps the
version, so the filter is deliberate belt-and-braces and needs its own test).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import (
    NotesScopeSyncAuthority,
    NotesSyncAuthorityError,
)

pytestmark = pytest.mark.unit

_USER = "test-user"


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "chachanotes.sqlite3", client_id=_USER)
    yield database
    database.close_connection()


def test_version_states_cover_active_tombstone_and_missing(
    db: CharactersRAGDB,
) -> None:
    db.add_note("Alive", "alive body", note_id="note-alive")
    db.add_note("Doomed", "doomed body", note_id="note-doomed")
    db.update_note("note-alive", {"content": "alive v2"}, 1)
    db.soft_delete_note("note-doomed", 1)

    states = db.get_note_version_states(
        ["note-alive", "note-doomed", "note-never-existed"]
    )

    assert states == {
        "note-alive": {"version": 2, "deleted": False},
        "note-doomed": {"version": 2, "deleted": True},
    }


def test_version_states_chunk_past_500_ids(db: CharactersRAGDB) -> None:
    db.add_note("First", "first", note_id="note-first")
    db.add_note("Last", "last", note_id="note-last")
    # Real ids in different 500-id chunks; 499 unknown ids between them.
    ids = ["note-first"] + [f"missing-{i}" for i in range(499)] + ["note-last"]

    states = db.get_note_version_states(ids)

    assert states == {
        "note-first": {"version": 1, "deleted": False},
        "note-last": {"version": 1, "deleted": False},
    }


def test_version_states_empty_input_reads_nothing(db: CharactersRAGDB) -> None:
    assert db.get_note_version_states([]) == {}


def test_interop_service_wrapper_passes_through(
    db: CharactersRAGDB, tmp_path: Path
) -> None:
    base_dir = tmp_path / "notes-base"
    base_dir.mkdir(mode=0o700)
    service = NotesInteropService(
        base_db_directory=base_dir,
        api_client_id="test-app",
        global_db_to_use=db,
    )
    db.add_note("Wrapped", "wrapped body", note_id="note-wrapped")

    states = service.get_note_version_states(_USER, ["note-wrapped"])

    assert states == {"note-wrapped": {"version": 1, "deleted": False}}


async def test_scope_service_prefers_the_bulk_projection(db: CharactersRAGDB) -> None:
    calls = {"bulk": 0, "single": 0}

    class _Backend:
        def get_note_version_states(self, _user_id, note_ids):
            calls["bulk"] += 1
            return db.get_note_version_states(note_ids)

        def get_note_by_id(self, _user_id, note_id):
            calls["single"] += 1
            return db.get_note_by_id(note_id)

    db.add_note("Bulk", "bulk body", note_id="note-bulk")
    service = NotesScopeService(local_notes_service=_Backend(), server_service=None)

    states = await service.get_note_version_states_for_sync(
        scope=ScopeType.LOCAL_NOTE, note_ids=["note-bulk"], user_id=_USER
    )

    assert states == {"note-bulk": {"version": 1, "deleted": False}}
    assert calls == {"bulk": 1, "single": 0}


async def test_scope_service_falls_back_per_note_without_the_projection(
    db: CharactersRAGDB,
) -> None:
    class _LegacyBackend:
        def get_note_by_id(self, _user_id, note_id):
            return db.get_note_by_id(note_id)

    db.add_note("Legacy", "legacy body", note_id="note-legacy")
    db.add_note("Gone", "gone body", note_id="note-gone")
    db.soft_delete_note("note-gone", 1)
    service = NotesScopeService(
        local_notes_service=_LegacyBackend(), server_service=None
    )

    states = await service.get_note_version_states_for_sync(
        scope=ScopeType.LOCAL_NOTE,
        note_ids=["note-legacy", "note-gone", "note-missing"],
        user_id=_USER,
    )

    # get_note_by_id filters tombstones, so deleted and missing are absent --
    # both are treated as "changed" by the caller, which is the safe answer.
    assert states == {"note-legacy": {"version": 1, "deleted": False}}


async def test_scope_service_rejects_non_local_scope(db: CharactersRAGDB) -> None:
    service = NotesScopeService(local_notes_service=object(), server_service=None)
    with pytest.raises(RuntimeError, match="server_contract_missing"):
        await service.get_note_version_states_for_sync(
            scope=ScopeType.SERVER_NOTE, note_ids=["note-x"], user_id=_USER
        )


class _DirectBackend:
    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def get_note_version_states(self, _user_id, note_ids):
        return self._db.get_note_version_states(note_ids)

    def get_note_by_id(self, _user_id, note_id):
        return self._db.get_note_by_id(note_id)


async def test_authority_excludes_tombstones_at_its_own_level(
    db: CharactersRAGDB,
) -> None:
    db.add_note("Alive", "alive body", note_id="note-alive")
    db.add_note("Doomed", "doomed body", note_id="note-doomed")
    db.soft_delete_note("note-doomed", 1)
    authority = NotesScopeSyncAuthority(
        NotesScopeService(
            local_notes_service=_DirectBackend(db), server_service=None
        ),
        scope=ScopeType.LOCAL_NOTE,
        user_id=_USER,
        note_scope_id="local_note",
    )

    versions = await authority.observe_versions(("note-alive", "note-doomed"))

    assert versions == {"note-alive": 1}


async def test_authority_wraps_backend_failure_into_bounded_reason(
    db: CharactersRAGDB,
) -> None:
    class _FailingBackend:
        def get_note_version_states(self, _user_id, note_ids):
            raise RuntimeError("private backend outage with details")

    authority = NotesScopeSyncAuthority(
        NotesScopeService(
            local_notes_service=_FailingBackend(), server_service=None
        ),
        scope=ScopeType.LOCAL_NOTE,
        user_id=_USER,
        note_scope_id="local_note",
    )

    with pytest.raises(NotesSyncAuthorityError, match="note_observation_failed"):
        await authority.observe_versions(("note-any",))
