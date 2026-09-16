"""The note -> file lookup the editor header asks (task-32640).

Real ``NotesDeviceStateStore`` on disk, real records. The runtime half is
exercised through a ``NotesSyncRuntimeOwner`` instance whose store and root
paths are the real ones, so the join between "which binding" and "which
root path" is the production join rather than a restatement of it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootRecord,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_models import NotesSyncSerializationProfile
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

_DIGEST = hashlib.sha256(b"note-location").hexdigest()
_PROFILE = NotesSyncSerializationProfile(
    utf8_bom=False, newline="lf", final_newline=True, mode=0o644
)


def _store_with_binding(
    tmp_path: Path,
    *,
    state: NotesSyncBindingState = NotesSyncBindingState.ACTIVE,
    relative_path: str = "People/Sam.md",
) -> tuple[NotesDeviceStateStore, Path]:
    root_folder = tmp_path / "vault"
    root_folder.mkdir()
    store = NotesDeviceStateStore(tmp_path / "state.sqlite3")
    store.initialize()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path=str(root_folder.resolve()),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path=relative_path,
            stable_identity_digest=_DIGEST,
            state=state,
            serialization=_PROFILE,
            content_digest=_DIGEST,
            note_version=1,
        )
    )
    return store, root_folder


def test_an_active_binding_names_its_root_and_path(tmp_path: Path) -> None:
    """task-32640 AC#2: the note side of the binding, which had no reader."""
    store, _ = _store_with_binding(tmp_path)
    try:
        assert store.active_binding_path_for_note("note-1") == (
            "root-1",
            "People/Sam.md",
        )
        assert store.active_binding_path_for_note("note-2") is None
    finally:
        store.close()


def test_a_binding_that_is_not_active_is_not_a_location(tmp_path: Path) -> None:
    """A candidate binding is a proposal, not a file the note lives in.

    Without the ``state = 'active'`` predicate this returns the candidate
    and the header claims a file for a note whose adoption was never
    applied -- which is the vacuity guard for the test above: the two
    differ only in the binding's state.
    """
    store, _ = _store_with_binding(tmp_path, state=NotesSyncBindingState.CANDIDATE)
    try:
        assert store.active_binding_path_for_note("note-1") is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_the_runtime_joins_the_binding_to_its_root_path(tmp_path: Path) -> None:
    """task-32640 AC#2: an absolute path, built from live runtime state."""
    store, root_folder = _store_with_binding(tmp_path)
    owner = NotesSyncRuntimeOwner.__new__(NotesSyncRuntimeOwner)
    owner._store = store
    owner._root_paths = {"root-1": str(root_folder.resolve())}
    try:
        assert await owner.note_file_location("note-1") == str(
            root_folder.resolve() / "People" / "Sam.md"
        )
        # A note with no binding is the database-only world, not an error.
        assert await owner.note_file_location("note-2") == ""
        # A root whose path the runtime has not loaded cannot be named.
        owner._root_paths = {}
        assert await owner.note_file_location("note-1") == ""
    finally:
        store.close()
