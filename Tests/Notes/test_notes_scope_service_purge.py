"""Delete forever through the Notes scope service, the lasting-sync gate and RAG.

TASK-34000.29 (N-29, part 8b; AC#9, rulings T8b-b and T8b-c). The owner
ruled "yes with confimration" on 2026-10-29.

- ``NotesScopeService.purge_note`` asks lasting sync whether the note is
  still bound to a folder at the moment it purges (never from a captured
  row), refuses a bound note with the guide's sentence, refuses when it
  cannot tell, refuses a live note, and otherwise purges through
  ``CharactersRAGDB.hard_delete_note`` -- a real database here.
- ``NotesSyncRuntimeOwner.note_sync_bound`` reads the device store: every
  binding state except ``disconnected`` (terminal) keeps the note in sync.
- ``submit_note_index_removal`` queues the purged note's semantic-index
  projection for removal, the Media delete hook's own guard.

Base (71f8d8a487): none of the three exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_device_state_store import NotesSyncBindingState
from tldw_chatbook.Notes.notes_scope_service import (
    PURGE_REFUSED_SYNC_BOUND,
    NotePurgeRefusedError,
    NotesScopeService,
    ScopeType,
)
from tldw_chatbook.RAG_Search import ingestion_indexing

USER = "user-1"


class _Runtime:
    """Lasting sync's answer, read at call time (a list of answers in turn)."""

    def __init__(self, *answers: bool | Exception) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    async def note_sync_bound(self, note_id: str) -> bool:
        self.asked.append(note_id)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Policy:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.actions.append(action_id)


class _Profile:
    def __init__(self, tmp_path: Path) -> None:
        self.db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="t8b-svc")
        self.policy = _Policy()
        self.service = NotesScopeService(
            local_notes_service=NotesInteropService(
                base_db_directory=tmp_path,
                api_client_id="t8b-svc",
                global_db_to_use=self.db,
            ),
            server_service=None,
            policy_enforcer=self.policy,
        )
        self.live_id = self.db.add_note("Live note", "still here")
        self.note_id = self.db.add_note("Gone note", "secret body")
        version = self.db.get_note_by_id(self.note_id)["version"]
        self.db.soft_delete_note(self.note_id, version)
        self.version = version + 1

    def state(self, note_id: str) -> int | None:
        row = self.db.get_connection().execute(
            "SELECT deleted FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return None if row is None else int(row[0])


@pytest.fixture
def removals(monkeypatch) -> list[tuple[str, object]]:
    """Record the semantic-index removals a purge submits."""
    submitted: list[tuple[str, object]] = []
    monkeypatch.setattr(
        ingestion_indexing,
        "submit_note_index_removal",
        lambda note_id, *, source_path=None: submitted.append((note_id, source_path))
        or True,
    )
    return submitted


async def _purge(profile: _Profile, runtime, note_id: str | None = None, **kwargs):
    return await profile.service.purge_note(
        scope=ScopeType.LOCAL_NOTE,
        note_id=note_id or profile.note_id,
        version=kwargs.pop("version", profile.version),
        user_id=USER,
        sync_runtime=runtime,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_purge_note_deletes_an_unbound_tombstone_for_good(
    tmp_path, removals
) -> None:
    profile = _Profile(tmp_path)
    runtime = _Runtime(False)

    assert await _purge(profile, runtime) is True

    assert profile.state(profile.note_id) is None
    assert profile.state(profile.live_id) == 0
    assert runtime.asked == [profile.note_id]
    # The delete authority, not a new one the policy registry lacks.
    assert profile.policy.actions == ["notes.delete.local"]
    assert [note_id for note_id, _path in removals] == [profile.note_id]


@pytest.mark.asyncio
async def test_purge_note_refuses_a_note_kept_in_sync_with_a_folder(
    tmp_path, removals
) -> None:
    """Ruling T8b-c: refuse, don't warn -- with the guide's own sentence."""
    profile = _Profile(tmp_path)

    with pytest.raises(NotePurgeRefusedError) as refused:
        await _purge(profile, _Runtime(True))

    assert refused.value.reason_code == "sync_bound"
    assert refused.value.user_message == PURGE_REFUSED_SYNC_BOUND
    assert PURGE_REFUSED_SYNC_BOUND == (
        "This note is kept in sync with a folder — remove it from the "
        "folder's sync first."
    )
    assert profile.state(profile.note_id) == 1
    assert removals == []


@pytest.mark.asyncio
async def test_purge_note_asks_lasting_sync_on_every_purge(tmp_path, removals) -> None:
    """The binding is read when the purge runs, never remembered.

    The same service and runtime: a note bound since the first ask is
    refused on the second, with nothing purged.
    """
    first = _Profile(tmp_path / "a")
    runtime = _Runtime(True)
    with pytest.raises(NotePurgeRefusedError):
        await _purge(first, runtime)
    runtime.answers = [False]
    assert await _purge(first, runtime) is True
    assert runtime.asked == [first.note_id, first.note_id]


@pytest.mark.asyncio
async def test_purge_note_refuses_when_lasting_sync_cannot_answer(
    tmp_path, removals
) -> None:
    """Fail closed: a check that cannot be made is never read as "not bound"."""
    profile = _Profile(tmp_path)

    with pytest.raises(NotePurgeRefusedError) as refused:
        await _purge(profile, _Runtime(RuntimeError("store busy")))

    assert refused.value.reason_code == "sync_unknown"
    assert "nothing was deleted" in refused.value.user_message
    assert profile.state(profile.note_id) == 1
    assert removals == []


@pytest.mark.asyncio
async def test_purge_note_refuses_a_live_note(tmp_path, removals) -> None:
    profile = _Profile(tmp_path)
    live_version = profile.db.get_note_by_id(profile.live_id)["version"]

    with pytest.raises(NotePurgeRefusedError) as refused:
        await _purge(profile, _Runtime(False), profile.live_id, version=live_version)

    assert refused.value.reason_code == "not_deleted"
    assert profile.state(profile.live_id) == 0
    assert removals == []


@pytest.mark.asyncio
async def test_purge_note_without_lasting_sync_in_the_app_still_purges(
    tmp_path, removals
) -> None:
    """No runtime at all: no folder can hold the note."""
    profile = _Profile(tmp_path)

    assert await _purge(profile, None) is True
    assert profile.state(profile.note_id) is None


@pytest.mark.asyncio
async def test_purge_note_is_local_only(tmp_path, removals) -> None:
    profile = _Profile(tmp_path)
    with pytest.raises(ValueError):
        await profile.service.purge_note(
            scope=ScopeType.SERVER_NOTE,
            note_id=profile.note_id,
            version=profile.version,
            user_id=USER,
        )
    assert profile.state(profile.note_id) == 1


# -- lasting sync's answer, from the real device store --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "bound"),
    [
        (NotesSyncBindingState.ACTIVE, True),
        (NotesSyncBindingState.CANDIDATE, True),
        (NotesSyncBindingState.PAUSED, True),
        (NotesSyncBindingState.NEEDS_ATTENTION, True),
        (NotesSyncBindingState.DISCONNECTED, False),
    ],
)
async def test_the_runtime_calls_every_binding_but_a_disconnected_one_bound(
    tmp_path, state, bound
) -> None:
    """Read live from the store, whether or not the root's path is loaded.

    ``note_file_location`` answers "" for a root whose path the runtime has
    not loaded yet (startup); that cannot be the purge's gate.
    """
    from Tests.Notes.test_notes_sync_note_location import _store_with_binding
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _owner

    store, _root = _store_with_binding(tmp_path, state=state)
    owner, _, _ = _owner(store=store, admitted=True, adapter=_Adapter([]))
    owner._root_paths = {}
    try:
        assert await owner.note_sync_bound("note-1") is bound
        assert await owner.note_sync_bound("note-2") is False
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_a_closing_runtime_refuses_to_say_a_note_is_unbound(tmp_path) -> None:
    from Tests.Notes.test_notes_sync_note_location import _store_with_binding
    from Tests.Notes.test_notes_sync_runtime import _Adapter, _owner

    store, _root = _store_with_binding(tmp_path)
    owner, _, _ = _owner(store=store, admitted=True, adapter=_Adapter([]))
    await owner.shutdown()

    with pytest.raises(RuntimeError):
        await owner.note_sync_bound("note-2")


# -- the semantic index -----------------------------------------------------


class _Indexer:
    def __init__(self) -> None:
        self.removals: list[object] = []

    def submit_removal(self, removal) -> bool:
        self.removals.append(removal)
        return True


def test_a_purged_note_is_queued_out_of_the_semantic_index(monkeypatch) -> None:
    indexer = _Indexer()
    monkeypatch.setattr(ingestion_indexing, "peek_shared_rag_service", lambda: None)
    monkeypatch.setattr(ingestion_indexing, "semantic_indexing_available", lambda: True)
    monkeypatch.setattr(ingestion_indexing, "get_ingestion_indexer", lambda: indexer)

    assert ingestion_indexing.submit_note_index_removal(
        "note-9", source_path=Path("/profile/notes.sqlite3")
    )

    (removal,) = indexer.removals
    assert removal.item_id == "note-9"
    assert removal.item_type == ingestion_indexing.ITEM_TYPE_NOTE
    assert removal.document_id == "note_note-9"
    assert removal.source_path == Path("/profile/notes.sqlite3")


def test_no_semantic_runtime_and_none_available_queues_nothing(monkeypatch) -> None:
    """The Media hook's guard: never initialize RAG just to delete."""
    indexer = _Indexer()
    monkeypatch.setattr(ingestion_indexing, "peek_shared_rag_service", lambda: None)
    monkeypatch.setattr(ingestion_indexing, "semantic_indexing_available", lambda: False)
    monkeypatch.setattr(ingestion_indexing, "get_ingestion_indexer", lambda: indexer)

    assert ingestion_indexing.submit_note_index_removal("note-9") is False
    assert indexer.removals == []
