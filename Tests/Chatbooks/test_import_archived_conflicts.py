"""Import conflict policies include archived, non-deleted conversations."""

import pytest

from Tests.Chatbooks.test_import_transactions import _build_synthetic_chatbook
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize(
    "resolution", [ConflictResolution.SKIP, ConflictResolution.RENAME]
)
def test_import_conflict_policy_and_unique_titles_include_archived_rows(
    tmp_path, resolution
):
    path = tmp_path / "archive-conflicts.db"
    db = CharactersRAGDB(path, "archive-import")
    title = "Synthetic Conversation 0"
    ids = [db.add_conversation({"title": name}) for name in (title, f"{title} (1)")]
    db.set_conversations_archived(
        ids,
        archived=True,
        expected_versions={
            cid: db.get_conversation_by_id(cid)["version"] for cid in ids
        },
    )
    originals = [db.get_conversation_by_id(cid) for cid in ids]
    status = ImportStatus()
    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(path)})
    success, _ = importer.import_chatbook(
        chatbook_path=_build_synthetic_chatbook(tmp_path),
        conflict_resolution=resolution,
        import_status=status,
    )
    assert success and status.failed_items == 0
    assert [db.get_conversation_by_id(cid) for cid in ids] == originals
    assert len(db.get_conversation_by_name(title, archive_scope="all")) == 1
    if resolution == ConflictResolution.SKIP:
        assert status.skipped_items == 1 and status.successful_items == 2
    else:
        assert status.skipped_items == 0 and status.successful_items == 3
        renamed = db.get_conversation_by_name(f"{title} (2)")
        assert len(renamed) == 1 and renamed[0]["id"] not in ids
    db.close_connection()
