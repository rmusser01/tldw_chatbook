"""Real SQLite archive lifecycle, optimistic Undo, and pre-pagination scope."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def db(tmp_path):
    store = CharactersRAGDB(tmp_path / "archive.db", client_id="archive-test")
    yield store
    store.close_connection()


def seed(db, title="Archive target"):
    cid = db.add_conversation({"title": title, "state": "resolved"})
    mid = db.add_message(
        {"conversation_id": cid, "sender": "user", "content": "Needle retained history"}
    )
    return cid, mid, db.get_conversation_by_id(cid)["version"]


def test_archive_restore_preserves_identity_messages_and_workflow(db):
    cid, mid, version = seed(db)
    service = ChatConversationService(db)
    result = service.set_conversations_archived(
        [cid, cid], archived=True, expected_versions={cid: version}
    )
    assert result == {"changed": {cid: version + 1}, "failures": {}}
    assert service.get_conversation_metadata(cid)["archived"] is True
    row = db.get_conversation_by_id(cid)
    assert row["deleted"] == 0 and row["state"] == "resolved"
    assert db.get_library_conversation_messages(cid)["messages"][0]["id"] == mid
    assert service.get_conversation_tree(cid)["conversation"]["id"] == cid
    db.close_connection()
    assert db.get_conversation_archive_states([cid, "absent"]) == {cid: True}
    assert service.set_conversations_archived(
        [cid], archived=False, expected_versions=result["changed"]
    ) == {"changed": {cid: version + 2}, "failures": {}}
    assert service.get_conversation_metadata(cid)["archived"] is False


def test_bulk_reports_only_actual_changes_and_stale_undo_refuses(db):
    first, _, first_v = seed(db, "First")
    stale, _, stale_v = seed(db, "Stale")
    missing_v, _, _ = seed(db, "No version")
    db.update_conversation(stale, {"title": "Intervening edit"}, stale_v)
    result = db.set_conversations_archived(
        [first, stale, missing_v, "absent"],
        archived=True,
        expected_versions={first: first_v, stale: stale_v, "absent": 1},
    )
    assert result == {
        "changed": {first: first_v + 1},
        "failures": {
            stale: "stale_version",
            missing_v: "missing_version",
            "absent": "not_found",
        },
    }
    assert db.set_conversations_archived(
        [first], archived=True, expected_versions=result["changed"]
    )["failures"] == {first: "already_archived"}
    db.update_conversation(first, {"title": "Edited while archived"}, first_v + 1)
    undo = db.set_conversations_archived(
        [first], archived=False, expected_versions=result["changed"]
    )
    assert undo == {"changed": {}, "failures": {first: "stale_version"}}
    assert db.get_conversation_archive_states([first, stale]) == {
        first: True,
        stale: False,
    }


@pytest.mark.parametrize("query", [None, "Needle"])
def test_archive_scope_filters_before_count_page_and_locator(db, query):
    service = ChatConversationService(db)
    ids = [seed(db, f"Item {i:02d}")[0] for i in range(45)]
    versions = {cid: db.get_conversation_by_id(cid)["version"] for cid in ids[::2]}
    service.set_conversations_archived(
        ids[::2], archived=True, expected_versions=versions
    )
    for scope, expected in [
        ("active", set(ids[1::2])),
        ("archived", set(ids[::2])),
        ("all", set(ids)),
    ]:
        payload = service.list_conversations(query, archive_scope=scope, limit=20)
        assert payload["pagination"]["total"] == len(expected)
        second = service.list_conversations(
            query, archive_scope=scope, limit=20, offset=20
        )
        third = service.list_conversations(
            query, archive_scope=scope, limit=20, offset=40
        )
        assert {
            item["id"] for page in [payload, second, third] for item in page["items"]
        } == expected
        target = second["items"][0]["id"]
        located = service.locate_conversation_page(target, query, archive_scope=scope)
        assert located["pagination"]["offset"] == 20
        assert located["pagination"]["total"] == len(expected)
        assert located["items"] == second["items"]
        library = service.search_library_conversations(
            query="Needle", archive_scope=scope, limit=100
        )
        assert library["total"] == len(expected)
        assert {item["id"] for item in library["items"]} == expected
        assert all(
            item["archived"] == (item["id"] in versions) for item in library["items"]
        )
        assert service.list_library_conversations(archive_scope=scope, limit=100)[
            "total"
        ] == len(expected)
    assert service.list_conversations()["pagination"]["total"] == 22
    assert service.locate_conversation_page(ids[0]) is None
    with pytest.raises((ValueError, InputError)):
        service.list_conversations(archive_scope="invalid")


def test_v70_migration_preserves_existing_rows_and_adds_archive_index(
    tmp_path, monkeypatch
):
    path = tmp_path / "v70.db"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 70)
        old = CharactersRAGDB(path, client_id="archive-test")
        cid, mid, version = seed(old)
        assert "archived" not in {
            r[1]
            for r in old.get_connection().execute("PRAGMA table_info(conversations)")
        }
        old.close_connection()
    executed = []
    execute = CharactersRAGDB._execute_migration_statements

    def record_artifact(self, cursor, script, label):
        if label == "V70→V71":
            executed.append(script)
        return execute(self, cursor, script, label)

    monkeypatch.setattr(
        CharactersRAGDB, "_execute_migration_statements", record_artifact
    )
    current = CharactersRAGDB(path, client_id="archive-test")
    artifact = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/DB/migrations/chachanotes_v70_to_v71_conversation_archive.sql"
    )
    assert executed == [artifact.read_text(encoding="utf-8")]
    assert current.get_conversation_by_id(cid)["archived"] == 0
    assert current.get_conversation_by_id(cid)["version"] == version
    assert current.get_library_conversation_messages(cid)["messages"][0]["id"] == mid
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 72
        )
        assert any(
            row[2] == "archived"
            for row in conn.execute("PRAGMA index_info(idx_conversations_archive)")
        )
    current.close_connection()


def test_legacy_discovery_defaults_active_and_explicit_scope_includes_archive(db):
    cid, _, version = seed(db)
    character_id = db.add_character_card({"name": "Archive persona"})
    db.update_conversation(cid, {"character_id": character_id}, version)
    row = db.get_conversation_by_id(cid)
    version = row["version"]
    keyword_id = db.add_keyword("archive-key")
    db.link_conversation_to_keyword(cid, keyword_id)
    db.set_conversations_archived(
        [cid], archived=True, expected_versions={cid: version}
    )
    readers = [
        lambda **kw: db.list_all_active_conversations(**kw),
        lambda **kw: db.get_conversation_by_name("Archive target", **kw),
        lambda **kw: db.get_conversations_for_character(row["character_id"], **kw),
        lambda **kw: db.search_conversations_by_title("Archive", **kw),
        lambda **kw: db.search_conversations_by_content("Needle", **kw),
        lambda **kw: db.get_conversations_for_keyword(keyword_id, **kw),
    ]
    for read in readers:
        assert read() == []
        assert [item["id"] for item in read(archive_scope="archived")] == [cid]


def test_archive_is_independent_of_deletion_and_rejects_non_boolean(db):
    cid, _, version = seed(db)
    db.set_conversations_archived(
        [cid], archived=True, expected_versions={cid: version}
    )
    db.soft_delete_conversation(cid, version + 1)
    current = db.get_conversation_by_id(cid, include_deleted=True)
    assert db.get_conversation_archive_states([cid]) == {}
    assert db.set_conversations_archived(
        [cid], archived=False, expected_versions={cid: current["version"]}
    ) == {"changed": {}, "failures": {cid: "not_found"}}
    db.restore_conversation(cid, current["version"])
    assert db.get_conversation_archive_states([cid]) == {cid: True}
    with pytest.raises(InputError):
        db.set_conversations_archived(
            [cid], archived="false", expected_versions={cid: version}
        )


def test_migration_failure_rolls_back_column_index_and_version(tmp_path, monkeypatch):
    path = tmp_path / "blocked-migration.db"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 70)
        old = CharactersRAGDB(path, client_id="archive-test")
        old.get_connection().execute("""CREATE TRIGGER block_archive_migration BEFORE UPDATE OF version ON db_schema_version
            WHEN NEW.version = 71 BEGIN SELECT RAISE(ABORT, 'blocked'); END""")
        old.get_connection().commit()
        old.close_connection()
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError

    with pytest.raises(SchemaError):
        CharactersRAGDB(path, client_id="archive-test")
    with sqlite3.connect(path) as conn:
        assert "archived" not in {
            r[1] for r in conn.execute("PRAGMA table_info(conversations)")
        }
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 70
        )
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='idx_conversations_archive'"
            ).fetchone()
            is None
        )


def test_competing_archive_writers_change_version_once(db):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cid, _, version = seed(db)
    start = threading.Barrier(2)

    def change():
        start.wait(timeout=5)
        try:
            return db.set_conversations_archived(
                [cid], archived=True, expected_versions={cid: version}
            )
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(change) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert sum(bool(result["changed"]) for result in results) == 1
    assert (
        sum(result["failures"].get(cid) == "stale_version" for result in results) == 1
    )
    assert db.get_conversation_by_id(cid)["version"] == version + 1


def test_archived_scope_combines_with_workspace_union_and_term_filters(db):
    service = ChatConversationService(db)
    first = db.add_conversation(
        {"title": "Design review", "scope_type": "workspace", "workspace_id": "work-a"}
    )
    second = db.add_conversation(
        {"title": "Design review", "scope_type": "workspace", "workspace_id": "work-b"}
    )
    versions = {
        cid: db.get_conversation_by_id(cid)["version"] for cid in [first, second]
    }
    db.set_conversations_archived(
        [first, second], archived=True, expected_versions=versions
    )
    payload = service.list_conversations(
        scope_type="all",
        archive_scope="archived",
        workspace_ids=["work-a"],
        query_terms=["Design"],
        query_workspace_ids_by_term=[[]],
        query_include_global_scope_by_term=[False],
        limit=1,
    )
    assert payload["pagination"]["total"] == 1
    assert [row["id"] for row in payload["items"]] == [first]
    assert (
        service.list_conversations(scope_type="all", workspace_ids=["work-a"])[
            "pagination"
        ]["total"]
        == 0
    )


def test_archived_character_seek_pagination_keeps_cursor_semantics(db):
    character = db.add_character_card({"name": "Seek persona"})
    ids = [
        db.add_conversation({"title": f"Chat {i}", "character_id": character})
        for i in range(4)
    ]
    versions = {cid: db.get_conversation_by_id(cid)["version"] for cid in ids[::2]}
    db.set_conversations_archived(ids[::2], archived=True, expected_versions=versions)
    with db.transaction() as cursor:
        for i, cid in enumerate(ids):
            cursor.execute(
                "UPDATE conversations SET last_modified = ? WHERE id = ?",
                (f"2026-09-10T12:00:0{i}.000Z", cid),
            )
    first = db.get_conversations_for_character(
        character, archive_scope="archived", limit=1
    )
    assert [row["id"] for row in first] == [ids[2]]
    second = db.get_conversations_for_character(
        character,
        archive_scope="archived",
        limit=1,
        before_last_modified=first[0]["last_modified"],
        before_id=first[0]["id"],
    )
    assert [row["id"] for row in second] == [ids[0]]


def test_archive_transitions_stay_local_and_receipts_reject_stale_cycle(db):
    cid, _, version = seed(db)
    db.update_conversation(cid, {"title": "Unsent shared title"}, version)
    version += 1
    connection = db.get_connection()
    before = [
        tuple(row)
        for row in connection.execute("SELECT * FROM sync_log ORDER BY change_id")
    ]
    for archived in (True, False, True):
        result = db.set_conversations_archived(
            [cid], archived=archived, expected_versions={cid: version}
        )
        version = result["changed"][cid]
        assert [
            tuple(row)
            for row in connection.execute("SELECT * FROM sync_log ORDER BY change_id")
        ] == before
        db.prune_sync_log()
        assert [
            tuple(row)
            for row in connection.execute("SELECT * FROM sync_log ORDER BY change_id")
        ] == before
    assert db.set_conversations_archived(
        [cid], archived=False, expected_versions={cid: version - 2}
    )["failures"] == {cid: "stale_version"}
    db.update_conversation(cid, {"title": "Still syncs while archived"}, version)
    payload = connection.execute(
        "SELECT payload FROM sync_log WHERE entity='conversations' AND entity_id=? "
        "ORDER BY change_id DESC LIMIT 1",
        (cid,),
    ).fetchone()[0]
    assert "Still syncs while archived" in payload
    assert "archived" not in json.loads(payload)
    version += 1
    db.soft_delete_conversation(cid, version)
    assert (
        connection.execute(
            "SELECT operation FROM sync_log WHERE entity='conversations' AND entity_id=? "
            "ORDER BY change_id DESC LIMIT 1",
            (cid,),
        ).fetchone()[0]
        == "delete"
    )
    db.restore_conversation(cid, version + 1)
    assert (
        connection.execute(
            "SELECT operation FROM sync_log WHERE entity='conversations' AND entity_id=? "
            "ORDER BY change_id DESC LIMIT 1",
            (cid,),
        ).fetchone()[0]
        == "update"
    )


def test_exact_title_lookup_is_bounded_and_paginates_after_archive_filter(db):
    ids = [db.add_conversation({"title": "Repeated"}) for _ in range(1005)]
    db.set_conversations_archived(
        ids[:3],
        archived=True,
        expected_versions={
            cid: db.get_conversation_by_id(cid)["version"] for cid in ids[:3]
        },
    )
    assert (
        len(db.get_conversation_by_name("Repeated", archive_scope="all", limit=5000))
        == 1000
    )
    first = db.get_conversation_by_name("Repeated")
    second = db.get_conversation_by_name("Repeated", limit=1000, offset=100)
    assert len(first) == 100
    assert len(second) == 902
    assert {row["id"] for row in first + second} == set(ids[3:])
    archived = db.get_conversation_by_name(
        "Repeated", archive_scope="archived", limit=2
    )
    archived_next = db.get_conversation_by_name(
        "Repeated", archive_scope="archived", limit=2, offset=2
    )
    assert {row["id"] for row in archived + archived_next} == set(ids[:3])
    assert len(archived) == 2 and len(archived_next) == 1
    for invalid in (0, -1, True, "2"):
        with pytest.raises(InputError):
            db.get_conversation_by_name("Repeated", limit=invalid)
    with pytest.raises(InputError):
        db.get_conversation_by_name("Repeated", offset=-1)


def test_mixed_archive_and_shared_payload_change_still_emits_sync_update(db):
    cid, _, version = seed(db)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET archived = 1, title = ?, version = version + 1 "
            "WHERE id = ?",
            ("Mixed change", cid),
        )
    row = (
        db.get_connection()
        .execute(
            "SELECT version, payload FROM sync_log WHERE entity='conversations' "
            "AND entity_id=? AND operation='update'",
            (cid,),
        )
        .fetchone()
    )
    assert row[0] == version + 1
    assert json.loads(row[1])["title"] == "Mixed change"
    assert "archived" not in json.loads(row[1])


@pytest.mark.parametrize("deleted", [False, True])
def test_conversation_retention_removes_late_superseded_payload(db, deleted):
    cid, _, version = seed(db)
    db.update_conversation(cid, {"title": "Current shared title"}, version)
    version += 1
    if deleted:
        db.soft_delete_conversation(cid, version)
    else:
        db.set_conversations_archived(
            [cid], archived=True, expected_versions={cid: version}
        )
    connection = db.get_connection()
    before = [
        tuple(row)
        for row in connection.execute("SELECT * FROM sync_log ORDER BY change_id")
    ]
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload) "
            "VALUES('conversations',?,'update',?,'late-client',1,?)",
            (
                cid,
                db._get_current_utc_timestamp_iso(),
                '{"title":"obsolete private text"}',
            ),
        )
    assert [
        tuple(row)
        for row in connection.execute("SELECT * FROM sync_log ORDER BY change_id")
    ] == before


@pytest.mark.parametrize("query", [None, "Needle"])
@pytest.mark.parametrize("deleted_only", [False, True])
def test_trash_includes_chats_archived_before_deletion(db, query, deleted_only):
    cid, _, version = seed(db)
    db.set_conversations_archived(
        [cid], archived=True, expected_versions={cid: version}
    )
    assert db.get_conversation_archive_states([cid]) == {cid: True}
    db.soft_delete_conversation(cid, version + 1)
    rows, total, _ = db.search_conversations_page(
        query, include_deleted=not deleted_only, deleted_only=deleted_only
    )
    assert total == 1 and [row["id"] for row in rows] == [cid]
