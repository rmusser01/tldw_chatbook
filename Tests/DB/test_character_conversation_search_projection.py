import importlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, local

import pytest

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
    CharacterKeywordIndexStatus,
    CharacterRepairRequest,
    CharacterRepairResult,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _card(db: CharactersRAGDB, name: str) -> int:
    card_id = db.add_character_card({"name": name})
    assert card_id is not None
    return card_id


def _chat(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    character_id: int,
    title: str,
    content: str,
    modified: str,
) -> None:
    authority = db.get_local_authority_id()
    assert (
        db.add_conversation(
            {
                "id": conversation_id,
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": title,
            }
        )
        == conversation_id
    )
    message_id = f"message-{conversation_id}"
    assert (
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": content,
                "timestamp": modified,
            }
        )
        == message_id
    )
    db.set_conversation_active_leaf(conversation_id, message_id)
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET last_modified = ? WHERE id = ?",
            (modified, conversation_id),
        )


def test_same_numeric_ids_in_two_authorities_never_merge(tmp_path: Path) -> None:
    first = CharactersRAGDB(tmp_path / "first.sqlite", client_id="first")
    second = CharactersRAGDB(tmp_path / "second.sqlite", client_id="second")
    try:
        _chat(
            first,
            conversation_id="same-id",
            character_id=1,
            title="First authority",
            content="first",
            modified="2026-09-03T10:00:00Z",
        )
        _chat(
            second,
            conversation_id="same-id",
            character_id=1,
            title="Second authority",
            content="second",
            modified="2026-09-03T10:00:00Z",
        )

        first_group = CharacterConversationNavigationService(first).recent_groups()[0]
        second_group = CharacterConversationNavigationService(second).recent_groups()[0]

        assert first_group.key != second_group.key
        assert first_group.rows[0].target != second_group.rows[0].target
        assert first_group.rows[0].title == "First authority"
        assert second_group.rows[0].title == "Second authority"
    finally:
        first.close_connection()
        second.close_connection()


def test_recent_groups_force_current_then_sort_other_groups_by_latest_chat(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "recent.sqlite", client_id="recent")
    current_id = _card(db, "Current but old")
    newest_id = _card(db, "Newest")
    middle_id = _card(db, "Middle")
    _chat(
        db,
        conversation_id="old-current",
        character_id=current_id,
        title="Current",
        content="current",
        modified="2026-09-01T10:00:00Z",
    )
    _chat(
        db,
        conversation_id="newest",
        character_id=newest_id,
        title="Newest",
        content="newest",
        modified="2026-09-03T10:00:00Z",
    )
    _chat(
        db,
        conversation_id="middle",
        character_id=middle_id,
        title="Middle",
        content="middle",
        modified="2026-09-02T10:00:00Z",
    )
    current = ResolvedLocalCharacterKey(db.get_local_authority_id(), current_id)

    groups = CharacterConversationNavigationService(
        db, current_character=current
    ).recent_groups(group_limit=3, row_limit=1)

    assert [group.character_label for group in groups] == [
        "Current but old",
        "Newest",
        "Middle",
    ]
    assert groups[0].is_current
    assert all(len(group.rows) == 1 and group.total == 1 for group in groups)


def test_recent_groups_force_include_valid_zero_chat_current_character(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "zero-current.sqlite", client_id="recent")
    current_id = _card(db, "Current without chats")
    other_id = _card(db, "Recent other")
    _chat(
        db,
        conversation_id="recent-other",
        character_id=other_id,
        title="Other",
        content="other",
        modified="2026-09-03T10:00:00Z",
    )

    groups = CharacterConversationNavigationService(
        db,
        current_character=ResolvedLocalCharacterKey(
            db.get_local_authority_id(), current_id
        ),
    ).recent_groups(group_limit=2)

    assert [group.character_label for group in groups] == [
        "Current without chats",
        "Recent other",
    ]
    assert groups[0].is_current
    assert groups[0].rows == ()
    assert groups[0].total == 0


def test_recent_groups_force_old_current_group_into_bound(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "old-current-bound.sqlite", client_id="recent")
    current_id = _card(db, "Old current")
    _chat(
        db,
        conversation_id="old-current",
        character_id=current_id,
        title="Old current",
        content="old",
        modified="2026-09-01T10:00:00Z",
    )
    for index in range(4):
        character_id = _card(db, f"Recent {index}")
        _chat(
            db,
            conversation_id=f"recent-{index}",
            character_id=character_id,
            title=f"Recent {index}",
            content="recent",
            modified=f"2026-09-03T10:00:0{index}Z",
        )

    groups = CharacterConversationNavigationService(
        db,
        current_character=ResolvedLocalCharacterKey(
            db.get_local_authority_id(), current_id
        ),
    ).recent_groups(group_limit=3, row_limit=1)

    assert [group.character_label for group in groups] == [
        "Old current",
        "Recent 3",
        "Recent 2",
    ]


def test_recent_groups_reserve_slot_for_nonempty_unavailable_group(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "unavailable-bound.sqlite", client_id="recent")
    for index in range(4):
        character_id = _card(db, f"Resolved {index}")
        _chat(
            db,
            conversation_id=f"resolved-{index}",
            character_id=character_id,
            title=f"Resolved {index}",
            content="resolved",
            modified=f"2026-09-03T10:00:0{index}Z",
        )
    _chat(
        db,
        conversation_id="unavailable",
        character_id=1,
        title="Unavailable",
        content="unavailable",
        modified="2026-09-03T09:00:00Z",
    )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET assistant_authority_id = NULL, "
            "assistant_id = 'unknown' WHERE id = 'unavailable'"
        )

    groups = CharacterConversationNavigationService(db).recent_groups(group_limit=4)

    assert [group.character_label for group in groups] == [
        "Resolved 3",
        "Resolved 2",
        "Resolved 1",
        "Chats with unavailable characters",
    ]
    assert groups[-1].total == 1
    assert groups[-1].rows[0].title == "Unavailable"


def test_unavailable_page_reports_complete_total_and_excludes_resolved(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "unavailable-page.sqlite", client_id="recent")
    character_id = _card(db, "Ada")
    for index, conversation_id in enumerate(("resolved", "missing-1", "missing-2")):
        _chat(
            db,
            conversation_id=conversation_id,
            character_id=character_id,
            title=conversation_id,
            content=conversation_id,
            modified=f"2026-09-03T10:00:0{index}Z",
        )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET assistant_authority_id = NULL "
            "WHERE id IN ('missing-1', 'missing-2')"
        )

    page = CharacterConversationNavigationService(db).unavailable_page(
        offset=0,
        limit=1,
    )

    assert page.total == 2
    assert [row.unresolved.conversation_id for row in page.rows] == ["missing-2"]
    assert all(row.target is None for row in page.rows)

    filtered = CharacterConversationNavigationService(db).unavailable_page(
        offset=0,
        limit=20,
        query="missing-1",
    )
    assert filtered.total == 1
    assert [row.unresolved.conversation_id for row in filtered.rows] == ["missing-1"]


def test_recent_groups_materialize_only_sql_bounded_sections(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "recent-sql-bound.sqlite", client_id="recent")
    for character_index in range(5):
        character_id = _card(db, f"Character {character_index}")
        for chat_index in range(3):
            _chat(
                db,
                conversation_id=f"chat-{character_index}-{chat_index}",
                character_id=character_id,
                title=f"Chat {character_index}-{chat_index}",
                content="bounded",
                modified=(f"2026-09-03T1{character_index}:00:0{chat_index}Z"),
            )
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    groups = CharacterConversationNavigationService(db).recent_groups(
        group_limit=2, row_limit=2
    )

    db.get_connection().set_trace_callback(None)
    assert len(groups) == 2
    assert all(group.total == 3 and len(group.rows) == 2 for group in groups)
    materializing_queries = [
        statement
        for statement in statements
        if "FROM conversations AS c" in statement and "COUNT(" not in statement
    ]
    assert materializing_queries
    assert all("LIMIT" in statement for statement in materializing_queries)
    section_queries = [
        statement for statement in statements if "GROUP BY c.character_id" in statement
    ]
    assert len(section_queries) == 1 and "LIMIT 2" in section_queries[0]
    plan = (
        db.get_connection()
        .execute(f"EXPLAIN QUERY PLAN {section_queries[0]}")
        .fetchall()
    )
    assert plan


def test_character_page_keyset_has_no_skip_or_repeat(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "paging.sqlite", client_id="paging")
    character_id = _card(db, "Paged")
    for index in range(5):
        _chat(
            db,
            conversation_id=f"chat-{index}",
            character_id=character_id,
            title=f"Chat {index}",
            content=f"body {index}",
            modified=f"2026-09-03T10:00:0{index}Z",
        )
    service = CharacterConversationNavigationService(db)
    key = ResolvedLocalCharacterKey(db.get_local_authority_id(), character_id)

    first = service.page_for_character(key, limit=2)
    second = service.page_for_character(key, cursor=first.next_cursor, limit=2)
    third = service.page_for_character(key, cursor=second.next_cursor, limit=2)
    row_ids = [
        row.target.conversation_id
        for page in (first, second, third)
        for row in page.rows
        if row.target is not None
    ]

    assert row_ids == ["chat-4", "chat-3", "chat-2", "chat-1", "chat-0"]
    assert len(set(row_ids)) == 5
    assert first.total == second.total == third.total == 5
    assert third.next_cursor is None


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("unavailable", [False, True])
def test_character_browse_uses_complete_date_key_without_skips_or_repeats(
    tmp_path: Path, limit: int, unavailable: bool
) -> None:
    db = CharactersRAGDB(tmp_path / "date-key.sqlite", client_id="date-key")
    character_id = _card(db, "Date ordering")
    # IDs deliberately oppose creation order; the newest pair also ties on creation.
    for conversation_id, created in (
        ("z-old", "2026-09-01T10:00:00Z"),
        ("b-new", "2026-09-03T10:00:00Z"),
        ("a-new", "2026-09-03T10:00:00Z"),
        ("y-middle", "2026-09-02T10:00:00Z"),
    ):
        _chat(
            db,
            conversation_id=conversation_id,
            character_id=character_id,
            title=conversation_id,
            content="date ordering",
            modified="2026-09-04T10:00:00Z",
        )
        with db.transaction() as connection:
            connection.execute(
                "UPDATE conversations SET created_at = ? WHERE id = ?",
                (created, conversation_id),
            )
    if unavailable:
        with db.transaction() as connection:
            connection.execute("UPDATE conversations SET assistant_authority_id = NULL")
    service = CharacterConversationNavigationService(db)
    key = ResolvedLocalCharacterKey(db.get_local_authority_id(), character_id)
    expected = ["b-new", "a-new", "y-middle", "z-old"]
    cursor = None
    seen = []
    for offset in range(0, 4, limit):
        page = (
            service.unavailable_page(offset=offset, limit=limit)
            if unavailable
            else service.page_for_character(key, cursor=cursor, limit=limit)
        )
        assert page.total == 4
        assert [row.title for row in page.rows] == expected[offset : offset + limit]
        seen.extend(row.title for row in page.rows)
        cursor = page.next_cursor
        if not unavailable and cursor is not None:
            assert cursor.created_at == page.rows[-1].created_at
    assert seen == expected
    assert len(set(seen)) == 4
    assert cursor is None
    assert [row.title for row in service.recent_groups(row_limit=4)[0].rows] == expected


def test_character_page_refresh_reselect_observes_order_key_mutation(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "date-refresh.sqlite", client_id="date-refresh")
    character_id = _card(db, "Refresh")
    for conversation_id in ("a", "b", "c"):
        _chat(
            db,
            conversation_id=conversation_id,
            character_id=character_id,
            title=conversation_id,
            content="refresh",
            modified="2026-09-04T10:00:00Z",
        )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET created_at = '2026-09-01T10:00:00Z'"
        )
    service = CharacterConversationNavigationService(db)
    key = ResolvedLocalCharacterKey(db.get_local_authority_id(), character_id)
    first = service.page_for_character(key, limit=1)
    assert [row.title for row in first.rows] == ["c"]
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET created_at = ? WHERE id = ?",
            ("2026-09-03T10:00:00Z", "a"),
        )
    continued = service.page_for_character(key, cursor=first.next_cursor, limit=1)
    assert [row.title for row in continued.rows] == ["b"]
    assert continued.next_cursor is None
    refreshed = service.page_for_character(key, limit=3)
    assert [row.title for row in refreshed.rows] == ["a", "c", "b"]
    assert refreshed.data_revision > first.data_revision


def test_keyword_search_is_local_only_and_revalidates_data_revision(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "keyword.sqlite", client_id="keyword")
    character_id = _card(db, "Keyword")
    _chat(
        db,
        conversation_id="local-chat",
        character_id=character_id,
        title="Local",
        content="LOCAL_KEYWORD_CANARY",
        modified="2026-09-03T10:00:00Z",
    )
    _chat(
        db,
        conversation_id="server-chat",
        character_id=character_id,
        title="Server",
        content="SERVER_KEYWORD_CANARY",
        modified="2026-09-03T11:00:00Z",
    )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET runtime_backend = 'server', "
            "discovery_owner = 'server', assistant_authority_id = 'server-authority' "
            "WHERE id = 'server-chat'"
        )
    service = CharacterConversationNavigationService(db)

    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    local = service.keyword_search("LOCAL_KEYWORD_CANARY")
    server = service.keyword_search("SERVER_KEYWORD_CANARY")
    assert local.keyword_status is CharacterKeywordIndexStatus.READY
    assert server.keyword_status is CharacterKeywordIndexStatus.READY
    assert [row.title for row in local.rows] == ["Local"]
    assert server.rows == ()

    db.increment_character_conversation_search_revision()
    stale = service.keyword_search("LOCAL_KEYWORD_CANARY")
    assert stale.rows == local.rows
    assert stale.data_revision == local.data_revision + 1
    assert stale.keyword_status is CharacterKeywordIndexStatus.READY
    assert stale.keyword_snapshot == local.keyword_snapshot
    assert stale.keyword_snapshot.source_revision == local.data_revision


def test_keyword_incrementally_reconciles_source_mutations(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "incremental.sqlite", client_id="incremental")
    original_card = _card(db, "Original character")
    replacement_card = _card(db, "Replacement character")
    _chat(
        db,
        conversation_id="maintained",
        character_id=original_card,
        title="Maintained",
        content="ORIGINAL_TERM",
        modified="2026-09-03T10:00:00Z",
    )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY

    assert (
        db.add_message(
            {
                "id": "appended",
                "conversation_id": "maintained",
                "parent_message_id": "message-maintained",
                "sender": "assistant",
                "role": "assistant",
                "content": "APPENDED_TERM",
            }
        )
        == "appended"
    )
    db.set_conversation_active_leaf("maintained", "appended")
    assert service.keyword_index_status() is CharacterKeywordIndexStatus.ABSENT
    pending = service.keyword_search("APPENDED_TERM")
    assert pending.keyword_status is CharacterKeywordIndexStatus.READY
    assert pending.rows == ()
    assert pending.keyword_snapshot.source_revision < pending.data_revision
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("APPENDED_TERM").total == 1

    assert db.update_message(
        "appended",
        {"content": "EDITED_TERM"},
        expected_version=1,
        preserve_descendants=True,
    )
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("APPENDED_TERM").total == 0
    assert service.keyword_search("EDITED_TERM").total == 1

    assert (
        db.add_message(
            {
                "id": "alternate",
                "conversation_id": "maintained",
                "parent_message_id": "message-maintained",
                "sender": "assistant",
                "role": "assistant",
                "content": "BRANCH_TERM",
            }
        )
        == "alternate"
    )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE messages SET variant_of = 'appended', "
            "is_selected_variant = CASE id WHEN 'alternate' THEN 1 ELSE 0 END "
            "WHERE id IN ('appended', 'alternate')"
        )
    db.set_conversation_active_leaf("maintained", "alternate")
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("EDITED_TERM").total == 0
    assert service.keyword_search("BRANCH_TERM").total == 1

    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET character_id = ?, assistant_id = ? "
            "WHERE id = 'maintained'",
            (replacement_card, str(replacement_card)),
        )
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("Replacement character").total == 1
    assert service.keyword_search("Original character").total == 0

    assert db.soft_delete_message("alternate", expected_version=1)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("BRANCH_TERM").total == 0
    plaintext = (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM character_conversation_search_documents "
            "WHERE body LIKE '%BRANCH_TERM%'"
        )
        .fetchone()[0]
    )
    assert plaintext == 0


def test_keyword_dirty_ledger_and_missed_event_reconciliation(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "reconcile.sqlite", client_id="reconcile")
    character_id = _card(db, "Reconcile")
    _chat(
        db,
        conversation_id="reconcile-chat",
        character_id=character_id,
        title="Reconcile",
        content="BEFORE_RECONCILE",
        modified="2026-09-03T10:00:00Z",
    )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert db.update_message(
        "message-reconcile-chat",
        {"content": "AFTER_RECONCILE"},
        expected_version=1,
        preserve_descendants=True,
    )
    with db.transaction() as connection:
        assert (
            connection.execute(
                "SELECT conversation_id FROM character_conversation_search_dirty"
            ).fetchone()[0]
            == "reconcile-chat"
        )
        connection.execute("DELETE FROM character_conversation_search_dirty")

    assert service.reconcile_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("BEFORE_RECONCILE").total == 0
    assert service.keyword_search("AFTER_RECONCILE").total == 1


def test_keyword_ensure_replaces_only_dirty_conversations(
    tmp_path: Path, monkeypatch
) -> None:
    db = CharactersRAGDB(tmp_path / "incremental-batch.sqlite", client_id="dirty")
    character_id = _card(db, "Dirty batch")
    for index in range(3):
        _chat(
            db,
            conversation_id=f"dirty-{index}",
            character_id=character_id,
            title=f"Dirty {index}",
            content="BEFORE_DIRTY",
            modified=f"2026-09-03T10:00:0{index}Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert db.update_message(
        "message-dirty-1",
        {"content": "AFTER_DIRTY"},
        expected_version=1,
        preserve_descendants=True,
    )
    projected: list[str] = []
    original_project = service._repository._projector.project

    def record_project(conversation_id: str, *, connection=None):
        projected.append(conversation_id)
        return original_project(conversation_id, connection=connection)

    monkeypatch.setattr(service._repository._projector, "project", record_project)

    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert projected == ["dirty-1"]
    assert service.keyword_search("AFTER_DIRTY").total == 1


def test_keyword_recovers_abandoned_build_and_cleans_superseded_rows(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "abandoned.sqlite", client_id="abandoned")
    character_id = _card(db, "Abandoned")
    _chat(
        db,
        conversation_id="abandoned-chat",
        character_id=character_id,
        title="Abandoned",
        content="RECOVERED_TERM",
        modified="2026-09-03T10:00:00Z",
    )
    revision = db.get_character_conversation_search_revision()
    authority = db.get_local_authority_id()
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO character_conversation_search_generations("
            "generation_id, data_authority_id, status, policy_version, "
            "source_revision, lease_expires_at) "
            "VALUES('abandoned', ?, 'building', 1, ?, '2000-01-01 00:00:00')",
            (authority, revision),
        )
    abandoned_service = CharacterConversationNavigationService(db)
    assert abandoned_service.keyword_index_status() is (
        CharacterKeywordIndexStatus.FAILED
    )
    assert abandoned_service.keyword_search("RECOVERED_TERM").keyword_status is (
        CharacterKeywordIndexStatus.FAILED
    )
    db.close_connection()
    restarted = CharactersRAGDB(
        tmp_path / "abandoned.sqlite", client_id="after-restart"
    )

    service = CharacterConversationNavigationService(restarted)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert service.keyword_search("RECOVERED_TERM").total == 1
    generations = (
        restarted.get_connection()
        .execute(
            "SELECT status, COUNT(*) FROM character_conversation_search_generations "
            "GROUP BY status"
        )
        .fetchall()
    )
    assert [(row["status"], row[1]) for row in generations] == [("ready", 1)]


def test_keyword_query_reports_active_build_instead_of_empty_success(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "building.sqlite", client_id="building")
    revision = db.get_character_conversation_search_revision()
    authority = db.get_local_authority_id()
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO character_conversation_search_generations("
            "generation_id, data_authority_id, status, policy_version, "
            "source_revision, lease_expires_at) "
            "VALUES('active', ?, 'building', 1, ?, DATETIME('now', '+5 minutes'))",
            (authority, revision),
        )
    service = CharacterConversationNavigationService(db)

    assert service.keyword_index_status() is CharacterKeywordIndexStatus.BUILDING
    page = service.keyword_search("anything")
    assert page.rows == () and page.total == 0
    assert page.keyword_status is CharacterKeywordIndexStatus.BUILDING


def test_keyword_search_fences_writer_during_snapshot_projection(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "projection-barrier.sqlite"
    reader = CharactersRAGDB(path, client_id="reader")
    character_id = _card(reader, "Barrier")
    _chat(
        reader,
        conversation_id="barrier-chat",
        character_id=character_id,
        title="Barrier",
        content="STALE_BARRIER_TERM",
        modified="2026-09-03T10:00:00Z",
    )
    service = CharacterConversationNavigationService(reader)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    writer = CharactersRAGDB(path, client_id="writer")
    original_project = service._repository._projector.project
    mutated = False

    def project_with_barrier(conversation_id: str, *, connection=None):
        nonlocal mutated
        if not mutated:
            mutated = True
            assert writer.update_message(
                "message-barrier-chat",
                {"content": "FRESH_BARRIER_TERM"},
                expected_version=1,
                preserve_descendants=True,
            )
        return original_project(conversation_id, connection=connection)

    monkeypatch.setattr(service._repository._projector, "project", project_with_barrier)
    try:
        result = service.keyword_search("STALE_BARRIER_TERM")
    finally:
        writer.close_connection()

    assert result.rows == ()
    assert result.total == 0
    assert result.keyword_status is CharacterKeywordIndexStatus.READY
    assert result.keyword_snapshot.source_revision < result.data_revision


def test_keyword_search_fences_writer_after_snapshot_before_return(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "final-fence.sqlite"
    reader = CharactersRAGDB(path, client_id="reader")
    character_id = _card(reader, "Final fence")
    _chat(
        reader,
        conversation_id="fence-chat",
        character_id=character_id,
        title="Fence",
        content="STALE_FINAL_TERM",
        modified="2026-09-03T10:00:00Z",
    )
    service = CharacterConversationNavigationService(reader)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    writer = CharactersRAGDB(path, client_id="writer")
    original_snapshot = service._repository._keyword_search_snapshot
    mutated = False

    def snapshot_then_mutate(query: str, *, offset: int, limit: int):
        nonlocal mutated
        result = original_snapshot(query, offset=offset, limit=limit)
        if not mutated:
            mutated = True
            assert writer.update_message(
                "message-fence-chat",
                {"content": "FRESH_FINAL_TERM"},
                expected_version=1,
                preserve_descendants=True,
            )
        return result

    monkeypatch.setattr(
        service._repository, "_keyword_search_snapshot", snapshot_then_mutate
    )
    try:
        result = service.keyword_search("STALE_FINAL_TERM")
    finally:
        writer.close_connection()

    assert result.rows == ()
    assert result.total == 0
    assert result.keyword_status is CharacterKeywordIndexStatus.READY
    assert result.keyword_snapshot.source_revision < result.data_revision


def test_keyword_search_refills_after_snapshot_revalidation_rejects_candidate(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "refill.sqlite", client_id="refill")
    character_id = _card(db, "Refill")
    for index in range(3):
        _chat(
            db,
            conversation_id=f"refill-{index}",
            character_id=character_id,
            title=f"Refill {index}",
            content="REFILL_TERM",
            modified=f"2026-09-03T10:00:0{index}Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    with db.transaction() as connection:
        connection.execute(
            "UPDATE character_conversation_search_documents "
            "SET eligibility_digest = 'corrupt' WHERE conversation_id = 'refill-2'"
        )

    result = service.keyword_search("REFILL_TERM", limit=2)

    assert [row.title for row in result.rows] == ["Refill 1", "Refill 0"]
    assert result.total == 2
    assert result.keyword_status is CharacterKeywordIndexStatus.READY


def test_keyword_exact_total_excludes_rejected_candidate_after_first_page(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "after-page.sqlite", client_id="after-page")
    character_id = _card(db, "After page")
    for index in range(60):
        _chat(
            db,
            conversation_id=f"after-page-{index:02d}",
            character_id=character_id,
            title=f"After page {index:02d}",
            content="WIDE_RESULT_TERM",
            modified=f"2026-09-03T10:00:{index:02d}Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    clean = service.keyword_search("WIDE_RESULT_TERM", limit=50)
    clean_tail = service.keyword_search("WIDE_RESULT_TERM", offset=50, limit=10)
    clean_ids = [
        row.target.conversation_id
        for row in (*clean.rows, *clean_tail.rows)
        if row.target is not None
    ]
    rejected_id = clean_ids[55]
    with db.transaction() as connection:
        connection.execute(
            "UPDATE character_conversation_search_documents "
            "SET eligibility_digest = 'corrupt' WHERE conversation_id = ?",
            (rejected_id,),
        )

    pages = [
        service.keyword_search("WIDE_RESULT_TERM", offset=offset, limit=20)
        for offset in (0, 20, 40)
    ]
    returned_ids = [
        row.target.conversation_id
        for page in pages
        for row in page.rows
        if row.target is not None
    ]

    assert all(
        page.keyword_status is CharacterKeywordIndexStatus.READY for page in pages
    )
    assert {page.total for page in pages} == {59}
    assert returned_ids == [item for item in clean_ids if item != rejected_id]
    assert len(returned_ids) == len(set(returned_ids)) == 59


def test_keyword_nonzero_offset_uses_validated_set_when_rejection_precedes_it(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "before-offset.sqlite", client_id="offset")
    character_id = _card(db, "Before offset")
    for index in range(60):
        _chat(
            db,
            conversation_id=f"before-offset-{index:02d}",
            character_id=character_id,
            title=f"Before offset {index:02d}",
            content="OFFSET_RESULT_TERM",
            modified=f"2026-09-03T10:00:{index:02d}Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    clean = service.keyword_search("OFFSET_RESULT_TERM", limit=50)
    clean_tail = service.keyword_search("OFFSET_RESULT_TERM", offset=50, limit=10)
    clean_ids = [
        row.target.conversation_id
        for row in (*clean.rows, *clean_tail.rows)
        if row.target is not None
    ]
    rejected_id = clean_ids[5]
    with db.transaction() as connection:
        connection.execute(
            "UPDATE character_conversation_search_documents "
            "SET eligibility_digest = 'corrupt' WHERE conversation_id = ?",
            (rejected_id,),
        )

    page = service.keyword_search("OFFSET_RESULT_TERM", offset=20, limit=20)
    page_ids = [
        row.target.conversation_id for row in page.rows if row.target is not None
    ]
    expected = [item for item in clean_ids if item != rejected_id][20:40]

    assert page.keyword_status is CharacterKeywordIndexStatus.READY
    assert page.total == 59
    assert page_ids == expected
    assert len(page_ids) == len(set(page_ids)) == 20


def test_keyword_search_indexes_character_display_identity(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "character-name.sqlite", client_id="keyword")
    character_id = _card(db, "DISPLAY_IDENTITY_CANARY")
    _chat(
        db,
        conversation_id="character-name-chat",
        character_id=character_id,
        title="Ordinary title",
        content="ordinary body",
        modified="2026-09-03T10:00:00Z",
    )
    service = CharacterConversationNavigationService(db)

    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    result = service.keyword_search("DISPLAY_IDENTITY_CANARY")

    assert [row.target.conversation_id for row in result.rows if row.target] == [
        "character-name-chat"
    ]


def test_keyword_search_applies_page_bound_inside_sqlite(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "keyword-bound.sqlite", client_id="keyword")
    character_id = _card(db, "Bounded")
    for index in range(6):
        _chat(
            db,
            conversation_id=f"bounded-{index}",
            character_id=character_id,
            title=f"Bounded {index}",
            content="BOUNDARY_CANARY",
            modified=f"2026-09-03T10:00:0{index}Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    result = service.keyword_search("BOUNDARY_CANARY", offset=2, limit=2)

    db.get_connection().set_trace_callback(None)
    assert result.total == 6
    assert len(result.rows) == 2
    candidate_queries = [
        statement
        for statement in statements
        if "bm25(character_conversation_fts)" in statement
    ]
    assert len(candidate_queries) == 1
    assert "LIMIT 2 OFFSET 2" in candidate_queries[0]


def test_unique_legacy_link_backfills_but_ambiguous_link_stays_unavailable(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "legacy.sqlite", client_id="legacy")
    character_id = _card(db, "Legacy")
    _chat(
        db,
        conversation_id="unique",
        character_id=character_id,
        title="Unique",
        content="unique",
        modified="2026-09-03T10:00:00Z",
    )
    _chat(
        db,
        conversation_id="ambiguous",
        character_id=character_id,
        title="Ambiguous",
        content="ambiguous",
        modified="2026-09-03T11:00:00Z",
    )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET assistant_authority_id = NULL "
            "WHERE id IN ('unique', 'ambiguous')"
        )
        connection.execute(
            "UPDATE conversations SET assistant_id = 'historical-unknown' "
            "WHERE id = 'ambiguous'"
        )

    assert db.backfill_character_conversation_legacy_links() == 1
    groups = CharacterConversationNavigationService(db).recent_groups()
    rows = {row.title: row for group in groups for row in group.rows}

    assert rows["Unique"].target is not None
    assert rows["Ambiguous"].unresolved == UnresolvedConversationKey(
        db.get_local_authority_id(), "ambiguous"
    )
    assert (
        rows["Ambiguous"].unavailable_reason
        is UnavailableCharacterReason.AMBIGUOUS_LEGACY_LINK
    )


def test_repair_candidates_stay_in_authority_and_repair_uses_expected_version(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "repair.sqlite", client_id="repair")
    replacement_id = _card(db, "Replacement")
    _chat(
        db,
        conversation_id="repair-me",
        character_id=1,
        title="Repair me",
        content="repair",
        modified="2026-09-03T10:00:00Z",
    )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET assistant_authority_id = NULL, "
            "assistant_id = 'unknown' WHERE id = 'repair-me'"
        )
    authority = db.get_local_authority_id()
    unresolved = UnresolvedConversationKey(authority, "repair-me")
    service = CharacterConversationNavigationService(db)

    _chat(
        db,
        conversation_id="already-resolved",
        character_id=replacement_id,
        title="Resolved",
        content="resolved",
        modified="2026-09-03T11:00:00Z",
    )
    assert (
        service.repair_candidates(
            UnresolvedConversationKey(authority, "already-resolved")
        )
        == ()
    )

    candidates = service.repair_candidates(unresolved)
    assert candidates
    assert {candidate.key.data_authority_id for candidate in candidates} == {authority}
    assert (
        service.repair_candidates(
            UnresolvedConversationKey("different-authority", "repair-me")
        )
        == ()
    )
    assert (
        service.repair(
            CharacterRepairRequest(
                unresolved=unresolved,
                replacement=ResolvedLocalCharacterKey(
                    "different-authority", replacement_id
                ),
                expected_conversation_version=1,
            )
        )
        is CharacterRepairResult.INVALID_CANDIDATE
    )
    assert (
        service.repair(
            CharacterRepairRequest(
                unresolved=unresolved,
                replacement=ResolvedLocalCharacterKey(authority, replacement_id),
                expected_conversation_version=99,
            )
        )
        is CharacterRepairResult.STALE_VERSION
    )

    before_revision = db.get_character_conversation_search_revision()
    assert (
        service.repair(
            CharacterRepairRequest(
                unresolved=unresolved,
                replacement=ResolvedLocalCharacterKey(authority, replacement_id),
                expected_conversation_version=1,
            )
        )
        is CharacterRepairResult.APPLIED
    )
    assert db.get_character_conversation_search_revision() == before_revision + 1
    repaired = db.get_conversation_by_id("repair-me")
    assert repaired is not None
    assert repaired["character_id"] == replacement_id
    assert repaired["assistant_authority_id"] == authority
    assert repaired["version"] == 2


def test_app_import_and_startup_leave_keyword_index_dormant(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        CharacterConversationNavigationService,
        "ensure_keyword_index",
        lambda self: calls.append("called"),
    )
    import tldw_chatbook.app as app_module

    importlib.reload(app_module)
    app_module.TldwCli()

    assert calls == []


def test_keyword_backfill_reports_each_128_conversation_batch(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "progress.sqlite", client_id="progress")
    character_id = _card(db, "Batch")
    for index in range(128):
        _chat(
            db,
            conversation_id=f"batch-{index:03d}",
            character_id=character_id,
            title=f"Batch {index}",
            content=f"content {index}",
            modified="2026-09-03T10:00:00Z",
        )
    observed: list[tuple[int, CharacterKeywordIndexStatus, int]] = []
    service: CharacterConversationNavigationService

    def record_progress(count: int) -> None:
        row = (
            db.get_connection()
            .execute(
                "SELECT processed_conversations "
                "FROM character_conversation_search_generations "
                "WHERE status = 'building'"
            )
            .fetchone()
        )
        assert row is not None
        observed.append((count, service.keyword_index_status(), int(row[0])))

    service = CharacterConversationNavigationService(
        db,
        progress_callback=record_progress,
    )

    assert service.keyword_index_status() is CharacterKeywordIndexStatus.ABSENT
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert observed == [(128, CharacterKeywordIndexStatus.BUILDING, 128)]


def test_keyword_backfill_streams_multiple_bounded_id_batches(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "streaming.sqlite", client_id="streaming")
    character_id = _card(db, "Streaming")
    for index in range(257):
        _chat(
            db,
            conversation_id=f"stream-{index:03d}",
            character_id=character_id,
            title=f"Stream {index}",
            content=f"content {index}",
            modified="2026-09-03T10:00:00Z",
        )
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    status = CharacterConversationNavigationService(db).ensure_keyword_index()

    db.get_connection().set_trace_callback(None)
    id_queries = [
        statement
        for statement in statements
        if "SELECT c.id" in statement and "assistant_authority_id" in statement
    ]
    assert status is CharacterKeywordIndexStatus.READY
    assert len(id_queries) == 3
    assert all("LIMIT 128" in statement for statement in id_queries)


def test_keyword_backfill_rejects_source_revision_change(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "revision-fence.sqlite", client_id="revision")
    character_id = _card(db, "Revision fence")
    for index in range(128):
        _chat(
            db,
            conversation_id=f"revision-{index:03d}",
            character_id=character_id,
            title=f"Revision {index}",
            content=f"content {index}",
            modified="2026-09-03T10:00:00Z",
        )
    service = CharacterConversationNavigationService(
        db,
        progress_callback=lambda _count: (
            db.increment_character_conversation_search_revision()
        ),
    )

    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.FAILED
    assert service.keyword_index_status() is CharacterKeywordIndexStatus.FAILED
    failed = service.keyword_search("content")
    assert failed.rows == ()
    assert failed.keyword_status is CharacterKeywordIndexStatus.FAILED


def test_explicit_keyword_retry_rebuilds_failed_current_generation(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "retry.sqlite", client_id="retry")
    character_id = _card(db, "Retry")
    for index in range(128):
        _chat(
            db,
            conversation_id=f"retry-{index:03d}",
            character_id=character_id,
            title=f"Retry {index}",
            content=f"retry content {index}",
            modified="2026-09-03T10:00:00Z",
        )

    def fail_progress(_count: int) -> None:
        raise RuntimeError("injected progress failure")

    assert (
        CharacterConversationNavigationService(
            db, progress_callback=fail_progress
        ).ensure_keyword_index()
        is CharacterKeywordIndexStatus.FAILED
    )

    retry = CharacterConversationNavigationService(db)
    assert retry.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert retry.keyword_search("retry content").total == 128


@pytest.mark.parametrize("change", ["deleted", "ineligible"])
@pytest.mark.parametrize("fail_replacement", [False, True])
def test_keyword_prior_snapshot_remains_queryable_during_replacement(
    tmp_path: Path, monkeypatch, change: str, fail_replacement: bool
) -> None:
    db = CharactersRAGDB(tmp_path / "prior-ready.sqlite", client_id="snapshot")
    character_id = _card(db, "Snapshot")
    for name in ("A", "B"):
        _chat(
            db,
            conversation_id=name,
            character_id=character_id,
            title=name,
            content="SNAPSHOT_TERM",
            modified="2026-09-03T10:00:00Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    original = service.keyword_search("SNAPSHOT_TERM")
    with db.transaction() as connection:
        if change == "deleted":
            connection.execute("UPDATE conversations SET deleted = 1 WHERE id = 'A'")
    if change == "ineligible":
        assert db.soft_delete_message("message-A", expected_version=1)
    advanced = service.keyword_search("SNAPSHOT_TERM")
    assert [row.title for row in advanced.rows] == ["B"]
    assert advanced.total == 1
    assert advanced.keyword_snapshot == original.keyword_snapshot
    assert advanced.keyword_snapshot.completed_at
    assert advanced.keyword_snapshot.source_revision < advanced.data_revision
    replacement = CharacterConversationNavigationService(db)
    monkeypatch.setattr(replacement._repository, "_POLICY_VERSION", 2)
    store = replacement._repository._replace_documents
    observed = []

    def hold_replacement(*args, **kwargs):
        page = replacement.keyword_search("SNAPSHOT_TERM")
        observed.append(page)
        assert [row.title for row in page.rows] == ["B"]
        assert page.keyword_snapshot == original.keyword_snapshot
        assert (
            replacement.keyword_index_status() is CharacterKeywordIndexStatus.BUILDING
        )
        if fail_replacement:
            raise RuntimeError("replacement failed")
        store(*args, **kwargs)

    monkeypatch.setattr(replacement._repository, "_replace_documents", hold_replacement)
    status = replacement.ensure_keyword_index()
    assert observed
    assert status is (
        CharacterKeywordIndexStatus.FAILED
        if fail_replacement
        else CharacterKeywordIndexStatus.READY
    )
    final = replacement.keyword_search("SNAPSHOT_TERM")
    assert [row.title for row in final.rows] == ["B"]
    if fail_replacement:
        assert final.keyword_snapshot == original.keyword_snapshot
    else:
        assert (
            final.keyword_snapshot.generation_id
            != original.keyword_snapshot.generation_id
        )
        assert final.keyword_snapshot.policy_version == 2
        assert final.keyword_snapshot.source_revision == final.data_revision


@pytest.mark.parametrize(
    "corruption", ["generation_revision", "document_revision", "timestamp"]
)
def test_keyword_rejects_inconsistent_snapshot_metadata(
    tmp_path: Path, corruption: str
) -> None:
    db = CharactersRAGDB(tmp_path / "snapshot-corrupt.sqlite", client_id="corrupt")
    card = _card(db, "Corrupt")
    _chat(
        db,
        conversation_id="A",
        character_id=card,
        title="A",
        content="CORRUPT_TERM",
        modified="2026-09-03T10:00:00Z",
    )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    with db.transaction() as connection:
        if corruption == "generation_revision":
            connection.execute(
                "UPDATE character_conversation_search_generations SET source_revision = source_revision + 1"
            )
        elif corruption == "document_revision":
            connection.execute(
                "UPDATE character_conversation_search_documents SET source_revision = source_revision + 1"
            )
        else:
            connection.execute(
                "UPDATE character_conversation_search_generations SET completed_at = NULL"
            )
    page = service.keyword_search("CORRUPT_TERM")
    assert page.rows == ()
    assert page.total == 0
    if corruption == "timestamp":
        assert page.keyword_status is CharacterKeywordIndexStatus.FAILED


def test_keyword_competing_builders_claim_one_sqlite_owner(
    tmp_path: Path, monkeypatch
) -> None:
    db = CharactersRAGDB(tmp_path / "builders.sqlite", client_id="builders")
    services = [CharacterConversationNavigationService(db) for _ in range(2)]
    transaction = db.transaction
    first_commits = Barrier(2)
    old_insert_commits = Barrier(2)
    counters = local()

    @contextmanager
    def interleaved_transaction(*args, **kwargs):
        counters.count = getattr(counters, "count", 0) + 1
        count = counters.count
        statements = []
        db.get_connection().set_trace_callback(statements.append)
        try:
            with transaction(*args, **kwargs) as connection:
                yield connection
        finally:
            db.get_connection().set_trace_callback(None)
        if count == 1:
            first_commits.wait(timeout=5)
        elif count == 2 and any(
            "INSERT INTO character_conversation_search_generations" in sql
            for sql in statements
        ):
            old_insert_commits.wait(timeout=5)

    monkeypatch.setattr(db, "transaction", interleaved_transaction)

    def build(service):
        try:
            return service.ensure_keyword_index()
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build, service) for service in services]
        statuses = [future.result(timeout=10) for future in futures]
    monkeypatch.setattr(db, "transaction", transaction)
    rows = (
        db.get_connection()
        .execute(
            "SELECT generation_id, status FROM character_conversation_search_generations"
        )
        .fetchall()
    )
    assert sorted(status.value for status in statuses) == ["building", "ready"]
    assert len(rows) == 1 and rows[0]["status"] == "ready"


@pytest.mark.parametrize("supersede", ["expired", "deleted"])
def test_keyword_superseded_builder_cannot_remove_new_ready_generation(
    tmp_path: Path, monkeypatch, supersede: str
) -> None:
    db = CharactersRAGDB(tmp_path / "superseded.sqlite", client_id="superseded")
    card = _card(db, "Superseded")
    _chat(
        db,
        conversation_id="A",
        character_id=card,
        title="A",
        content="SURVIVING_TERM",
        modified="2026-09-03T10:00:00Z",
    )
    old = CharacterConversationNavigationService(db)
    replacement = CharacterConversationNavigationService(db)
    store = old._repository._replace_documents
    ready_ids = []

    def supersede_after_store(generation_id, *args, **kwargs):
        store(generation_id, *args, **kwargs)
        with db.transaction() as connection:
            if supersede == "expired":
                connection.execute(
                    "UPDATE character_conversation_search_generations "
                    "SET lease_expires_at = '2000-01-01' WHERE generation_id = ?",
                    (generation_id,),
                )
            else:
                connection.execute(
                    "DELETE FROM character_conversation_search_generations WHERE generation_id = ?",
                    (generation_id,),
                )
        assert replacement.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
        ready_ids.append(
            db.get_connection()
            .execute(
                "SELECT generation_id FROM character_conversation_search_generations WHERE status = 'ready'"
            )
            .fetchone()[0]
        )

    monkeypatch.setattr(old._repository, "_replace_documents", supersede_after_store)
    assert old.ensure_keyword_index() is CharacterKeywordIndexStatus.FAILED
    remaining = (
        db.get_connection()
        .execute(
            "SELECT generation_id FROM character_conversation_search_generations WHERE status = 'ready'"
        )
        .fetchall()
    )
    assert [row[0] for row in remaining] == ready_ids
    assert replacement.keyword_search("SURVIVING_TERM").total == 1


def test_keyword_metadata_updates_do_not_reindex_unrelated_text(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "fts-work.sqlite", client_id="fts-work")
    card = _card(db, "FTS work")
    for name in ("A", "B", "C"):
        _chat(
            db,
            conversation_id=name,
            character_id=card,
            title=name,
            content="BEFORE_TERM",
            modified="2026-09-03T10:00:00Z",
        )
    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    reindexed = []
    connection = db.get_connection()
    connection.create_function("record_fts_update", 1, reindexed.append)
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'character_conversation_search_documents_au'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER character_conversation_search_documents_au")
    connection.execute(
        trigger.replace(
            "BEGIN", "BEGIN SELECT record_fts_update(new.conversation_id);", 1
        )
    )
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET last_modified = '2026-09-04' WHERE id = 'A'"
        )
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert reindexed == []
    assert db.update_message(
        "message-A",
        {"content": "AFTER_TERM"},
        expected_version=1,
        preserve_descendants=True,
    )
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert reindexed == ["A"]
    assert service.keyword_search("AFTER_TERM").total == 1
    assert service.keyword_search("BEFORE_TERM").total == 2
