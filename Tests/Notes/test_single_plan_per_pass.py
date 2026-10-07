"""B11: the reconciliation plan is computed once per pass, off the event loop.

``observe_root`` used to run the full :func:`plan_reconciliation` per pass
only to read ``observation_token`` (on the event loop), and the caller then
planned the same observations again -- 2 plan executions and 3 token hashes
per pass, one of them blocking the loop. One pass must plan exactly once,
on a worker thread, and the observation token must stay byte-identical to
the value the old derivation produced.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

import tldw_chatbook.Notes.notes_sync_runtime as runtime_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    _observation_token,
    plan_reconciliation,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    _ProductionRuntimeAdapter,
    build_notes_sync_runtime_owner,
)

pytestmark = pytest.mark.unit

_USER = "test-user"
_CONTENT = "same body\n"


class _DbBackedLocalNotes:
    """The production local-notes call surface over a real DB."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def get_note_by_id(self, _user_id: str, note_id: str):
        return self._db.get_note_by_id(note_id)

    def get_note_version_states(self, _user_id: str, note_ids):
        return self._db.get_note_version_states(note_ids)

    def update_note(self, _user_id: str, note_id: str, data, expected_version: int):
        return self._db.update_note(note_id, data, expected_version)

    def soft_delete_note(self, _user_id: str, note_id: str, version: int):
        return self._db.soft_delete_note(note_id, version)

    def add_note(self, _user_id: str, title: str, content: str, note_id=None):
        return self._db.add_note(title, content, note_id=note_id)


@dataclass
class _World:
    db: CharactersRAGDB
    store: NotesDeviceStateStore
    adapter: _ProductionRuntimeAdapter
    owner: object


@pytest.fixture()
def world(tmp_path: Path) -> _World:
    root_dir = (tmp_path / "sync-root").resolve()
    root_dir.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "chachanotes.sqlite3", client_id=_USER)
    service = NotesScopeService(
        _DbBackedLocalNotes(db),
        None,
        folder_repository=LocalNoteFolderRepository(db),
    )
    store = NotesDeviceStateStore(tmp_path / "device-state.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root = NotesSyncRootRecord(
        root_id="root-1",
        note_scope_id="local_note",
        logical_folder_id="folder-1",
        canonical_path=str(root_dir),
        direction=NotesSyncDirection.BIDIRECTIONAL,
        state=NotesSyncRootState.ACTIVE,
    )
    store.create_root(root)

    (root_dir / "note.md").write_text(_CONTENT, encoding="utf-8")
    with PosixNotesSyncFilesystem(root_dir) as filesystem:
        file = filesystem.observe("note.md")
    db.add_note("Note", _CONTENT, note_id="note-1")
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
            state=NotesSyncBindingState.ACTIVE,
            serialization=file.observation.serialization,
            content_digest=hashlib.sha256(_CONTENT.encode("utf-8")).hexdigest(),
            note_version=1,
        )
    )

    owner = build_notes_sync_runtime_owner(
        notes_scope_service=service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        # Same store file the fixture seeded: the owner opens its own
        # connection to the rooted state.
        database_path=tmp_path / "device-state.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id=_USER,
        recovery_capacity_bytes=1024 * 1024,
    )
    built = _World(db=db, store=store, adapter=owner._adapter, owner=owner)
    yield built
    # The tests shut the owner down (which also closes the adapter); the
    # fixture only owns the database handle.
    db.close_connection()


async def test_one_plan_and_one_token_per_pass_off_the_event_loop(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    await world.owner.start()
    try:
        counts = {"plan": 0, "token": 0}
        plan_threads: list[threading.Thread] = []
        real_plan = runtime_module.plan_reconciliation
        real_token = runtime_module._observation_token

        def counting_plan(request):
            counts["plan"] += 1
            plan_threads.append(threading.current_thread())
            return real_plan(request)

        def counting_token(request):
            counts["token"] += 1
            return real_token(request)

        monkeypatch.setattr(runtime_module, "plan_reconciliation", counting_plan)
        monkeypatch.setattr(runtime_module, "_observation_token", counting_token)

        plan = await world.owner.check_root("root-1")

        assert counts["plan"] == 1, f"plan executed {counts['plan']}x per pass"
        assert counts["token"] == 1, f"token derived {counts['token']}x per pass"
        loop_thread = threading.current_thread()
        assert plan_threads, "the pass never planned"
        assert all(thread is not loop_thread for thread in plan_threads), (
            "the plan ran on the event loop thread"
        )
        # Unchanged world: the pass classifies the binding as no_change.
        assert [action.kind for action in plan.safe_actions] == [
            NotesSyncActionKind.NO_CHANGE
        ]
        assert not plan.attention and not plan.skips
    finally:
        await world.owner.shutdown()


async def test_pass_token_is_byte_identical_to_the_observation_derivation(
    world: _World,
) -> None:
    """The token that feeds change detection must equal what the pre-change
    derivation produced: ``_observation_token`` over the same observations,
    which is exactly what ``plan_reconciliation(...).observation_token``
    always carried."""
    await world.owner.start()
    try:
        plan = await world.owner.check_root("root-1")

        # Reference derivation over a fresh observation of the same state
        # (a warm pass returns observations equal to a cold one).
        root = world.store.get_root("root-1")
        request = await world.adapter.observe_root(root)
        reference = plan_reconciliation(request).observation_token
        world.adapter.release_observation(reference)

        assert plan.observation_token == reference
        assert plan.observation_token == _observation_token(request)
    finally:
        await world.owner.shutdown()
