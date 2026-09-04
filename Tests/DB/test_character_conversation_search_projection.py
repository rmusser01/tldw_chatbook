import importlib
from pathlib import Path

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
    assert db.add_conversation(
        {
            "id": conversation_id,
            "character_id": character_id,
            "assistant_kind": "character",
            "assistant_id": str(character_id),
            "assistant_authority_id": authority,
            "title": title,
        }
    ) == conversation_id
    message_id = f"message-{conversation_id}"
    assert db.add_message(
        {
            "id": message_id,
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": content,
            "timestamp": modified,
        }
    ) == message_id
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
    assert [row.title for row in local.rows] == ["Local"]
    assert server.rows == ()

    db.increment_character_conversation_search_revision()
    stale = service.keyword_search("LOCAL_KEYWORD_CANARY")
    assert stale.rows == ()
    assert stale.data_revision == local.data_revision + 1


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
    assert service.repair_candidates(
        UnresolvedConversationKey(authority, "already-resolved")
    ) == ()

    candidates = service.repair_candidates(unresolved)
    assert candidates
    assert {candidate.key.data_authority_id for candidate in candidates} == {authority}
    assert service.repair_candidates(
        UnresolvedConversationKey("different-authority", "repair-me")
    ) == ()
    assert service.repair(
        CharacterRepairRequest(
            unresolved=unresolved,
            replacement=ResolvedLocalCharacterKey("different-authority", replacement_id),
            expected_conversation_version=1,
        )
    ) is CharacterRepairResult.INVALID_CANDIDATE
    assert service.repair(
        CharacterRepairRequest(
            unresolved=unresolved,
            replacement=ResolvedLocalCharacterKey(authority, replacement_id),
            expected_conversation_version=99,
        )
    ) is CharacterRepairResult.STALE_VERSION

    before_revision = db.get_character_conversation_search_revision()
    assert service.repair(
        CharacterRepairRequest(
            unresolved=unresolved,
            replacement=ResolvedLocalCharacterKey(authority, replacement_id),
            expected_conversation_version=1,
        )
    ) is CharacterRepairResult.APPLIED
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
        row = db.get_connection().execute(
            "SELECT processed_conversations "
            "FROM character_conversation_search_generations "
            "WHERE status = 'building'"
        ).fetchone()
        assert row is not None
        observed.append((count, service.keyword_index_status(), int(row[0])))

    service = CharacterConversationNavigationService(
        db,
        progress_callback=record_progress,
    )

    assert service.keyword_index_status() is CharacterKeywordIndexStatus.ABSENT
    assert service.ensure_keyword_index() is CharacterKeywordIndexStatus.READY
    assert observed == [(128, CharacterKeywordIndexStatus.BUILDING, 128)]


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
    assert service.keyword_search("content").rows == ()


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
