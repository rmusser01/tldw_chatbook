"""Real-store archive chronology and lifecycle revision regressions."""

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def test_workspace_cycle_uses_revision_not_clock(tmp_path):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    registry = LocalWorkspaceRegistryService(
        db, now_factory=lambda: "2026-09-20T12:00:00+00:00"
    )
    try:
        registry.create_workspace(workspace_id="w", name="Recovered")
        first = registry.archive_workspace("w")
        registry.unarchive_workspace("w")
        second = registry.archive_workspace("w")
        assert second.archived_at == first.archived_at
        assert second.archive_revision == first.archive_revision + 2
    finally:
        db.close()


@pytest.mark.parametrize(
    "later", ["2026-09-20T12:00:00+00:00", "2026-09-19T12:00:00+00:00"]
)
def test_stale_workspace_episode_is_rejected_with_repeating_or_backward_clock(
    tmp_path, later
):
    from tldw_chatbook.Workspaces.registry_service import WorkspaceArchiveConflict

    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    clock = ["2026-09-20T12:00:00+00:00"]
    registry = LocalWorkspaceRegistryService(db, now_factory=lambda: clock[0])
    try:
        registry.create_workspace(workspace_id="w", name="Recovered")
        first = registry.archive_workspace("w")
        registry.unarchive_workspace("w")
        clock[0] = later
        second = registry.archive_workspace("w")
        assert second.archived_at == later
        with pytest.raises(WorkspaceArchiveConflict):
            registry.unarchive_workspace(
                "w",
                expected_archive_revision=first.archive_revision,
                expected_name=first.name,
            )
        assert registry.get_workspace("w") == second
    finally:
        db.close()


def test_archive_time_does_not_advance_past_future_modification(tmp_path):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    now = "2026-09-20T12:00:00+00:00"
    registry = LocalWorkspaceRegistryService(db, now_factory=lambda: now)
    try:
        registry.create_workspace(workspace_id="w", name="Recovered")
        with db.transaction() as conn:
            conn.execute(
                "UPDATE workspace_records SET updated_at = ? WHERE workspace_id = ?",
                ("2099-01-01T00:00:00+00:00", "w"),
            )
        assert registry.archive_workspace("w").archived_at == now
    finally:
        db.close()


def test_conversation_chronology_is_local_and_uses_observed_clock(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "chats.sqlite", client_id="chronology")
    try:
        cid = db.add_conversation({"title": "Retained"})
        before = db.get_conversation_by_id(cid)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE conversations SET last_modified = ? WHERE id = ?",
                ("2099-01-01T00:00:00+00:00", cid),
            )
        now = "2026-09-20T12:00:00+00:00"
        monkeypatch.setattr(db, "_get_current_utc_timestamp_iso", lambda: now)
        with db.transaction() as conn:
            events = conn.execute(
                "SELECT * FROM sync_log WHERE entity_id = ?", (cid,)
            ).fetchall()
        result = db.set_conversations_archived(
            [cid], archived=True, expected_versions={cid: before["version"]}
        )
        archived = ChatConversationService(db).get_conversation_metadata(cid)
        assert archived["archived_at"] == now
        assert (
            db.list_library_conversations_page(
                limit=10, offset=0, archive_scope="archived"
            )["items"][0]["archived_at"]
            == now
        )
        assert db.set_conversations_archived(
            [cid], archived=True, expected_versions=result["changed"]
        )["failures"] == {cid: "already_archived"}
        assert db.get_conversation_by_id(cid)["archived_at"] == now
        with db.transaction() as conn:
            conn.execute(
                "UPDATE conversations SET archived_at = ?, version = version + 1 WHERE id = ?",
                ("2026-09-19T00:00:00+00:00", cid),
            )
            assert [
                tuple(row)
                for row in conn.execute(
                    "SELECT * FROM sync_log WHERE entity_id = ?", (cid,)
                )
            ] == [tuple(row) for row in events]
        version = db.get_conversation_by_id(cid)["version"]
        restored = db.set_conversations_archived(
            [cid], archived=False, expected_versions={cid: version}
        )
        assert restored["changed"] == {cid: version + 1}
        assert db.get_conversation_by_id(cid)["archived_at"] is None
        with db.transaction() as conn:
            assert [
                tuple(row)
                for row in conn.execute(
                    "SELECT * FROM sync_log WHERE entity_id = ?", (cid,)
                )
            ] == [tuple(row) for row in events]
        db.update_conversation(cid, {"title": "Shared edit"}, version + 1)
        with db.transaction() as conn:
            import json

            payload = json.loads(
                conn.execute(
                    "SELECT payload FROM sync_log WHERE entity_id = ?", (cid,)
                ).fetchone()[0]
            )
            assert payload["title"] == "Shared edit"
            assert "archived_at" not in payload and "archived" not in payload
    finally:
        db.close_connection()


def test_archive_chronology_index_orders_mixed_inventory_without_temp_sort(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "inventory.sqlite", client_id="chronology")
    try:
        with db.transaction() as conn:
            conn.executemany(
                "INSERT INTO conversations(id, root_id, client_id, title, deleted, archived, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        f"c-{index}",
                        f"c-{index}",
                        "chronology",
                        f"Chat {index}",
                        int(index % 5 == 0),
                        int(index % 2 == 0),
                        f"2026-09-{index % 27 + 1:02d}T12:00:00+00:00"
                        if index % 3
                        else None,
                    )
                    for index in range(500)
                ],
            )
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
                ).fetchone()
                is None
            )
            plan = conn.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM conversations WHERE deleted=0 AND archived=1 ORDER BY archived_at DESC, id LIMIT 20"
            ).fetchall()
            detail = " ".join(str(tuple(row)) for row in plan)
            assert "idx_conversations_archive_chronology" in detail
            assert "TEMP B-TREE" not in detail
    finally:
        db.close_connection()
