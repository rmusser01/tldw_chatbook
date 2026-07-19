"""Real-seam integration coverage for the Library notes sync panel.

Exercises the exact call the Library screen makes
(``NotesSyncService.sync_folder`` -> ``NotesSyncEngine.sync``) against a
real temp ChaChaNotes DB and a real ``tmp_path`` folder -- no fakes -- so
these tests prove the wiring the screen relies on, not just the pure
state helpers.
"""

import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.sync_engine import ConflictResolution, SyncDirection
from tldw_chatbook.Notes.sync_service import NotesSyncService

USER_ID = "library-sync-user"


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_db(temp_dir):
    db_path = temp_dir / "test_notes.db"
    db = CharactersRAGDB(str(db_path), "library_sync_test_client")
    yield db
    db.close_connection()


@pytest.fixture
def notes_service(test_db, temp_dir):
    service = NotesInteropService(
        base_db_directory=temp_dir,
        api_client_id="library_sync_test_api",
        global_db_to_use=test_db,
    )
    yield service
    service.close_all_user_connections()


@pytest.fixture
def sync_service(notes_service, test_db):
    # Mirrors exactly how the Library screen builds this
    # (``_run_library_notes_sync`` in library_screen.py):
    # NotesSyncService(notes_service=..., db=...).
    return NotesSyncService(notes_service=notes_service, db=test_db)


@pytest.mark.asyncio
async def test_library_notes_sync_db_to_disk_writes_matching_file(
    sync_service, notes_service, temp_dir
):
    """A note seeded in the DB (with no on-disk counterpart yet) lands as a
    file with matching content after a DB_TO_DISK sync -- the same call
    shape (``sync_folder(root_folder=..., user_id=..., direction=...,
    conflict_resolution=...)``) the Library screen's Sync-now button makes.
    """
    sync_dir = temp_dir / "sync_root"
    sync_dir.mkdir()

    note_id = notes_service.add_note(
        user_id=USER_ID,
        title="Library DB Note",
        content="Body written from the database side.",
    )
    note = notes_service.get_note_by_id(USER_ID, note_id)
    notes_service.update_note_sync_metadata(
        user_id=USER_ID,
        note_id=note_id,
        sync_metadata={
            "file_path_on_disk": str(sync_dir / "library_db_note.md"),
            "relative_file_path_on_disk": "library_db_note.md",
            "sync_root_folder": str(sync_dir),
            "is_externally_synced": 1,
            "file_extension": ".md",
        },
        expected_version=note["version"],
    )

    session_id, progress = await sync_service.sync_folder(
        root_folder=sync_dir,
        user_id=USER_ID,
        direction=SyncDirection.DB_TO_DISK,
        conflict_resolution=ConflictResolution.ASK,
    )

    assert session_id
    assert progress.errors == []
    written = sync_dir / "library_db_note.md"
    assert written.exists()
    assert written.read_text() == "Body written from the database side."


@pytest.mark.asyncio
async def test_library_notes_sync_disk_to_db_creates_note(
    sync_service, notes_service, temp_dir
):
    """A ``.md`` file dropped into the sync folder (with no DB counterpart)
    becomes a note after a DISK_TO_DB sync."""
    sync_dir = temp_dir / "sync_root"
    sync_dir.mkdir()
    (sync_dir / "dropped_note.md").write_text("Body written from disk.")

    session_id, progress = await sync_service.sync_folder(
        root_folder=sync_dir,
        user_id=USER_ID,
        direction=SyncDirection.DISK_TO_DB,
        conflict_resolution=ConflictResolution.ASK,
    )

    assert session_id
    assert progress.errors == []
    assert len(progress.created_notes) == 1
    notes = notes_service.list_notes(USER_ID)
    assert len(notes) == 1
    assert notes[0]["title"] == "dropped_note"
    assert notes[0]["content"] == "Body written from disk."


@pytest.mark.asyncio
async def test_library_notes_sync_bidirectional_newer_wins_disk_edit_wins(
    sync_service, notes_service, temp_dir
):
    """When both sides changed since the last sync and conflict_resolution
    is NEWER_WINS, the side with the more recent modification timestamp
    wins -- here, a disk edit made after the DB edit."""
    sync_dir = temp_dir / "sync_root"
    sync_dir.mkdir()

    note_id = notes_service.add_note(
        user_id=USER_ID,
        title="Conflict Note",
        content="Original content",
    )
    file_path = sync_dir / "conflict_note.md"
    note = notes_service.get_note_by_id(USER_ID, note_id)
    notes_service.update_note_sync_metadata(
        user_id=USER_ID,
        note_id=note_id,
        sync_metadata={
            "file_path_on_disk": str(file_path),
            "relative_file_path_on_disk": "conflict_note.md",
            "sync_root_folder": str(sync_dir),
            "is_externally_synced": 1,
            "file_extension": ".md",
        },
        expected_version=note["version"],
    )
    file_path.write_text("Original content")

    # Establish the baseline the engine diffs against (same shape the
    # engine itself uses in ``_update_note_sync_metadata`` -- see
    # test_sync_engine.py's own conflict test for this exact pattern).
    file_info = sync_service.sync_engine._get_file_info(file_path, sync_dir)
    notes_service.update_note_sync_metadata(
        user_id=USER_ID,
        note_id=note_id,
        sync_metadata={
            "last_synced_disk_file_hash": file_info.content_hash,
            "last_synced_disk_file_mtime": file_info.mtime,
        },
        expected_version=1,
    )

    # Change the DB side first (older last_modified)...
    notes_service.update_note(
        user_id=USER_ID,
        note_id=note_id,
        update_data={"content": "Modified in database"},
        expected_version=2,
    )

    # ...then change disk and force its mtime strictly later than the DB's
    # last_modified timestamp, so NEWER_WINS deterministically picks disk.
    file_path.write_text("Modified on disk")
    future_mtime = time.time() + 5
    import os

    os.utime(file_path, (future_mtime, future_mtime))

    session_id, progress = await sync_service.sync_folder(
        root_folder=sync_dir,
        user_id=USER_ID,
        direction=SyncDirection.BIDIRECTIONAL,
        conflict_resolution=ConflictResolution.NEWER_WINS,
    )

    assert session_id
    assert len(progress.conflicts) == 1
    conflict = progress.conflicts[0]
    assert conflict.conflict_type == "both_changed"
    assert conflict.db_content == "Modified in database"
    assert conflict.disk_content == "Modified on disk"

    # Disk was newer, so the DB note must now carry the disk's content.
    resolved_note = notes_service.get_note_by_id(USER_ID, note_id)
    assert resolved_note["content"] == "Modified on disk"


@pytest.mark.asyncio
async def test_library_notes_sync_progress_callback_reports_completion(
    sync_service, temp_dir
):
    """The ``progress_callback`` kwarg the screen threads through actually
    fires with the real engine, ending at processed == total -- the same
    contract the screen's marshaled UI update relies on."""
    sync_dir = temp_dir / "sync_root"
    sync_dir.mkdir()
    for index in range(3):
        (sync_dir / f"note{index}.md").write_text(f"Note {index}")

    updates = []

    def progress_callback(sync_progress):
        updates.append((sync_progress.processed_files, sync_progress.total_files))

    session_id, progress = await sync_service.sync_folder(
        root_folder=sync_dir,
        user_id=USER_ID,
        direction=SyncDirection.DISK_TO_DB,
        conflict_resolution=ConflictResolution.ASK,
        progress_callback=progress_callback,
    )

    assert session_id
    assert updates
    assert updates[-1] == (3, 3)
    assert progress.processed_files == 3
