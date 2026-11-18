"""Original local Notes import effects under the elected Windows profile writer."""

import sys
from uuid import uuid4

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
@private_profile_test
@pytest.mark.asyncio
async def test_local_note_import_effect_preserves_edges_keywords_retry_and_folder(request, monkeypatch):
    from tldw_chatbook.config import get_chachanotes_db_path
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.note_folder_models import FolderConflictError
    from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
    from tldw_chatbook.Notes.note_import_executor import ImportTargetConflictError, LocalNoteImportTarget
    from tldw_chatbook.Notes.note_import_plan_models import ParsedNotePayload

    db = CharactersRAGDB(get_chachanotes_db_path(), "file-import-user")
    folders = LocalNoteFolderRepository(db)
    target = LocalNoteImportTarget(db=db, folder_repository=folders)
    note_id = str(uuid4())
    folder_id = str(uuid4())

    def unexpected_local_mutation(*args, **kwargs):
        raise AssertionError("Original Notes import SQL ran in the client")

    try:
        peer = db.add_note("Wikilink peer", "Related")
        with monkeypatch.context() as patch:
            for name in ("_insert_note", "_update_note", "_sync_normalized_keywords"):
                patch.setattr(LocalNoteImportTarget, name, unexpected_local_mutation)
            patch.setattr(LocalNoteFolderRepository, "_attach_manual_with_cursor", unexpected_local_mutation)
            created = ParsedNotePayload(title="Imported", content="Linked [[Wikilink peer]]", keywords=("Project",))
            first = target.create_note(note_id=note_id, payload=created)
            assert first == target.create_note(note_id=note_id, payload=created)
            assert db.get_connection().execute(
                "SELECT COUNT(*) FROM note_links WHERE source_note_id = ? AND target_note_id = ?", (note_id, peer),
            ).fetchone()[0] == 1
            updated = ParsedNotePayload(title="Updated", content="No link", keywords=("edited",))
            changed = target.replace_note(note_id=note_id, expected_version=first.version, payload=updated)
            assert changed == target.replace_note(note_id=note_id, expected_version=first.version, payload=updated)
            with pytest.raises(ImportTargetConflictError):
                target.replace_note(note_id=note_id, expected_version=first.version, payload=created)
            target.sync_keywords(note_id=note_id, keywords=(key for key in ("red", "blue")))
            assert target.keywords_match(note_id=note_id, keywords=("red", "blue"))
            folder = target.ensure_folder(segments=(part for part in ("Imported",)), folder_id=folder_id, allow_existing=False)
            assert folder.folder_id == folder_id
            target.attach_membership(folder_id=folder_id, note_id=note_id, expected_note_version=changed.version)
        assert folders.get_exact_manual_membership(folder_id=folder_id, note_id=note_id) is not None
        assert db.get_connection().execute("SELECT COUNT(*) FROM note_links WHERE source_note_id = ?", (note_id,)).fetchone()[0] == 0
        current = folders.get_folder(folder_id)
        renamed = folders.rename_folder(folder_id, name="Renamed", expected_version=current.version)
        assert renamed.folder.name == "Renamed"
        with pytest.raises(FolderConflictError) as conflict:
            folders.rename_folder(folder_id, name="Stale", expected_version=current.version)
        assert conflict.value.reason == "folder_version"
        with pytest.raises(RuntimeError, match="rollback"):
            with db.transaction(immediate=True) as cursor:
                folders.rename_folder(folder_id, name="Rolled back", expected_version=renamed.folder.version, cursor=cursor)
                raise RuntimeError("rollback")
        assert folders.get_folder(folder_id) == renamed.folder
    finally:
        db.close_connection()
