"""Manage sync folders names every root (TASK-32451).

The path-free ``NotesSyncRootRuntimeSnapshot`` used to carry no name at all,
so the Library row for every root read "Sync folder (name unavailable before
cutover)" -- including a root the user had just typed a name for. These pins
run the REAL runtime (``build_notes_sync_runtime_owner`` over a disposable
``CharactersRAGDB`` + state store + vault directory) and read the published
projection: the name the user typed reaches the snapshot (AC#1), two roots
are told apart from the snapshot alone (AC#2), no vault path reaches the
snapshot, the row or a log record (AC#3), and a root without a folder reads
an honest fallback rather than a placeholder (AC#4).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRootSetup,
    NotesSyncRuntimeOwner,
    build_notes_sync_runtime_owner,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

MIGRATED_FALLBACK = "Migrated notes — review to finish setup"
SETTING_UP_FALLBACK = "Sync folder (setting up)"


def _seed_marker(state_path: Path) -> None:
    store = NotesDeviceStateStore(state_path)
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    store.close()


def _build_owner(tmp_path: Path, database: CharactersRAGDB) -> NotesSyncRuntimeOwner:
    """The production owner, as the app builds it, over disposable authorities."""

    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="task-32451",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=LocalNoteFolderRepository(database),
    )
    return build_notes_sync_runtime_owner(
        notes_scope_service=scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )


def _setup(vault: Path, display_name: str) -> NotesSyncRootSetup:
    vault.mkdir(parents=True, exist_ok=True)
    return NotesSyncRootSetup(
        display_name=display_name,
        canonical_path=str(vault.resolve()),
        note_scope_id="local_note",
        direction=NotesSyncDirection.BIDIRECTIONAL,
    )


async def _activate(owner: NotesSyncRuntimeOwner, vault: Path, display_name: str) -> str:
    """Set a root up the way the Library's Add-folder flow does; return its id."""

    review = await owner.review_setup(_setup(vault, display_name))
    result = await owner.activate_root(review.root_id, review.observation_token)
    assert result.accepted, (result.status, result.next_action)
    return review.root_id


def _names(owner: NotesSyncRuntimeOwner) -> dict[str, str]:
    return {root.root_id: root.display_name for root in owner.snapshot().roots}


def _controller(owner: NotesSyncRuntimeOwner) -> LibraryNotesSyncController:
    return LibraryNotesSyncController(
        runtime=owner,
        import_controller=SimpleNamespace(begin_selection=lambda: None),
    )


@pytest.fixture
def database(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="task-32451")
    _seed_marker(tmp_path / "sync.sqlite3")
    yield db
    db.close_connection()


async def test_published_snapshot_carries_the_folder_name_the_user_typed(
    tmp_path: Path, database: CharactersRAGDB
) -> None:
    """AC#1: the name typed at setup is the name the projection publishes."""

    owner = _build_owner(tmp_path, database)
    await owner.start()
    try:
        root_id = await _activate(owner, tmp_path / "vault-7f3a", "Vault sync")
        (root,) = owner.snapshot().roots
        assert root.root_id == root_id
        assert root.display_name == "Vault sync"
        # The same name the Notes tree shows for the managed folder.
        record = NotesDeviceStateStore(tmp_path / "sync.sqlite3").get_root(root_id)
        folder = LocalNoteFolderRepository(database).get_folder(
            record.logical_folder_id
        )
        assert folder is not None and folder.name == "Vault sync"
    finally:
        await owner.shutdown()


async def test_two_roots_are_distinguishable_from_the_snapshot_alone(
    tmp_path: Path, database: CharactersRAGDB
) -> None:
    """AC#2: two roots carry two different names, each its own folder's."""

    owner = _build_owner(tmp_path, database)
    await owner.start()
    try:
        first = await _activate(owner, tmp_path / "vault-a", "Vault sync")
        second = await _activate(owner, tmp_path / "vault-b", "Work notes")
        names = _names(owner)
        assert names == {first: "Vault sync", second: "Work notes"}
        store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
        folders = LocalNoteFolderRepository(database)
        for root_id, name in names.items():
            folder = folders.get_folder(store.get_root(root_id).logical_folder_id)
            assert folder is not None and folder.name == name
    finally:
        await owner.shutdown()


async def test_restart_republishes_names_from_the_store(
    tmp_path: Path, database: CharactersRAGDB
) -> None:
    """A fresh owner over the same state names the roots without any setup."""

    owner = _build_owner(tmp_path, database)
    await owner.start()
    try:
        first = await _activate(owner, tmp_path / "vault-a", "Vault sync")
        second = await _activate(owner, tmp_path / "vault-b", "Work notes")
    finally:
        await owner.shutdown()

    reopened = _build_owner(tmp_path, database)
    await reopened.start()
    try:
        assert _names(reopened) == {first: "Vault sync", second: "Work notes"}
    finally:
        await reopened.shutdown()


async def test_migrated_candidate_without_a_folder_reads_an_honest_fallback(
    tmp_path: Path, database: CharactersRAGDB
) -> None:
    """AC#4: no user-typed name yet -> say what the root IS, not that a name
    is "unavailable"."""

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    for root_id, state, code in (
        ("legacy-1", NotesSyncRootState.PAUSED, "migration_review_required"),
        ("pending-1", NotesSyncRootState.PENDING, None),
    ):
        vault = tmp_path / root_id
        vault.mkdir()
        store.create_root(
            NotesSyncRootRecord(
                root_id=root_id,
                note_scope_id="local_note",
                logical_folder_id=None,
                canonical_path=str(vault.resolve()),
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=state,
                last_status_code=code,
            )
        )
    store.close()

    owner = _build_owner(tmp_path, database)
    await owner.start()
    try:
        names = _names(owner)
        assert names["legacy-1"] == MIGRATED_FALLBACK
        # Startup publishes nothing for a PENDING root; Check changes on it
        # does (and refuses), which is the one route that lists it.
        with pytest.raises(RuntimeError, match="sync_root_not_active"):
            await owner.check_root("pending-1")
        assert _names(owner)["pending-1"] == SETTING_UP_FALLBACK
        assert "unavailable" not in " ".join(_names(owner).values())
    finally:
        await owner.shutdown()


async def test_no_path_reaches_the_snapshot_or_the_row(
    tmp_path: Path, database: CharactersRAGDB
) -> None:
    """AC#3: the name route carries a label, never the vault path."""

    vault = tmp_path / "vault-7f3a"
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="TRACE")
    owner = _build_owner(tmp_path, database)
    try:
        await owner.start()
        try:
            await _activate(owner, vault, "Vault sync")
            controller = _controller(owner)
            for root in owner.snapshot().roots:
                row = controller._project_root(root)
                for label in (root.display_name, row.display_name):
                    assert str(vault) not in label
                    assert str(vault.resolve()) not in label
                    assert os.sep not in label
                assert row.display_name == "Vault sync"
        finally:
            await owner.shutdown()
    finally:
        logger.remove(sink_id)
    leaked = [line for line in records if "vault-7f3a" in line]
    assert leaked == []


async def test_deleted_folder_still_names_the_root(
    tmp_path: Path, database: CharactersRAGDB
) -> None:
    """A soft-deleted managed folder keeps naming its root: the user still
    recognises it, and the status line is what says something is wrong."""

    owner = _build_owner(tmp_path, database)
    await owner.start()
    try:
        root_id = await _activate(owner, tmp_path / "vault-a", "Vault sync")
    finally:
        await owner.shutdown()
    record = NotesDeviceStateStore(tmp_path / "sync.sqlite3").get_root(root_id)
    folders = LocalNoteFolderRepository(database)
    folder = folders.get_folder(record.logical_folder_id)
    assert folder is not None
    folders.soft_delete_folder(folder.folder_id, expected_version=folder.version)
    assert folders.get_folder(folder.folder_id) is None

    reopened = _build_owner(tmp_path, database)
    await reopened.start()
    try:
        assert _names(reopened)[root_id] == "Vault sync"
    finally:
        await reopened.shutdown()


def test_snapshot_rejects_a_path_shaped_display_name() -> None:
    """The projection validates the name the way the setup form does."""

    for bad in ("a/b", "a\\b", "two\nlines", "x" * 161):
        with pytest.raises(ValueError):
            NotesSyncRootRuntimeSnapshot(
                "root-1", "up_to_date", "sync_now", display_name=bad
            )
    with pytest.raises(TypeError):
        NotesSyncRootRuntimeSnapshot(
            "root-1", "up_to_date", "sync_now", display_name=None  # type: ignore[arg-type]
        )
    # A markup-shaped name is a legal folder name and stays a plain label.
    snapshot = NotesSyncRootRuntimeSnapshot(
        "root-1", "up_to_date", "sync_now", display_name="Notes [2026] [@click=app.quit]"
    )
    assert snapshot.display_name == "Notes [2026] [@click=app.quit]"
    # Positional construction (every existing test) still works, name empty.
    assert NotesSyncRootRuntimeSnapshot("root-1", "up_to_date", "sync_now").display_name == ""
