"""Behavior tests for portable Notes organization materialization."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator

import pytest

from tldw_chatbook.Notes import notes_organization_repository as organization_repository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
    NotesOrganizationRepositoryError,
    portable_collision_key,
    portable_relative_path,
)
from tldw_chatbook.Sync_Interop.notes_organization import organization_link_id
from tldw_chatbook.Sync_Interop.notes_organization import (
    NotesOrganizationValidationError,
)


@pytest.fixture
def repository() -> Iterator[NotesOrganizationRepository]:
    db = CharactersRAGDB(":memory:", client_id="organization-tests")
    yield NotesOrganizationRepository(db, server_profile_id="profile-a")
    db.close_connection()


def _id(number: int) -> str:
    return str(uuid.UUID(f"00000000-0000-4000-8000-{number:012d}"))


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _apply(
    repository: NotesOrganizationRepository,
    *,
    domain: str,
    object_id: str,
    payload: dict[str, object],
    revision: int = 1,
    operation: str = "upsert",
    restore_intent: bool = False,
    base_server_cursor: str | None = None,
    base_object_revision: int | None = None,
    base_object_hash: str | None = None,
):
    with repository.db.transaction() as cursor:
        if restore_intent and base_object_revision is None and base_object_hash is None:
            head = cursor.execute(
                "SELECT server_cursor, object_revision, object_hash "
                "FROM notes_organization_heads WHERE server_profile_id = ? "
                "AND dataset_id = 'dataset-a' AND domain = ? AND object_id = ?",
                (repository.server_profile_id, domain, object_id),
            ).fetchone()
            if head is not None:
                base_server_cursor = str(head["server_cursor"])
                base_object_revision = int(head["object_revision"])
                base_object_hash = str(head["object_hash"])
        return repository.apply_envelope(
            cursor,
            dataset_id="dataset-a",
            domain=domain,
            object_id=object_id,
            operation=operation,
            payload=payload,
            object_revision=revision,
            object_hash=_hash(payload),
            server_cursor=f"cursor-{revision}",
            base_server_cursor=base_server_cursor,
            base_object_revision=base_object_revision,
            base_object_hash=base_object_hash,
            restore_intent=restore_intent,
        )


def test_portable_collision_rules_are_not_local_nfkc_rules() -> None:
    assert portable_collision_key("A") == "a"
    assert portable_collision_key("Ａ") == "ａ"
    assert portable_collision_key("A") != portable_collision_key("Ａ")
    assert portable_relative_path(("Parent", "Child")) == "parent/child"

    for invalid in (".", "..", "a/b", "a\\b"):
        with pytest.raises(NotesOrganizationRepositoryError):
            portable_collision_key(invalid)

    with pytest.raises(NotesOrganizationRepositoryError):
        portable_relative_path(("a" * 250, "b" * 250))


def test_resources_materialize_parent_first_and_lookup_by_sync_id(
    repository: NotesOrganizationRepository,
) -> None:
    parent_id, child_id = _id(1), _id(2)

    blocked = _apply(
        repository,
        domain="notes.folder",
        object_id=child_id,
        payload={"name": "Child", "parent_sync_id": parent_id},
    )
    assert blocked.status == "blocked"
    assert blocked.reason_code == "missing_parent"

    assert (
        _apply(
            repository,
            domain="notes.folder",
            object_id=parent_id,
            payload={"name": "Parent", "parent_sync_id": None},
        ).status
        == "applied"
    )
    assert (
        _apply(
            repository,
            domain="notes.folder",
            object_id=child_id,
            payload={"name": "Child", "parent_sync_id": parent_id},
            revision=2,
        ).status
        == "applied"
    )

    folder = repository.get_resource_by_sync_id("notes.folder", child_id)
    assert folder is not None
    assert folder["path"] == "/Parent/Child"
    assert folder["sync_id"] == child_id
    assert folder["id"] != child_id
    assert (
        repository.get_resource_by_sync_id("notes.folder", child_id)["id"]
        == folder["id"]
    )


def test_folder_hierarchy_cycle_is_blocked_without_mutating_projection(
    repository: NotesOrganizationRepository,
) -> None:
    parent_id, child_id = _id(3), _id(4)
    _apply(
        repository,
        domain="notes.folder",
        object_id=parent_id,
        payload={"name": "Parent", "parent_sync_id": None},
    )
    _apply(
        repository,
        domain="notes.folder",
        object_id=child_id,
        payload={"name": "Child", "parent_sync_id": parent_id},
    )

    result = _apply(
        repository,
        domain="notes.folder",
        object_id=parent_id,
        payload={"name": "Parent", "parent_sync_id": child_id},
        revision=2,
    )

    assert result.status == "blocked"
    assert result.reason_code == "hierarchy_cycle"
    assert (
        repository.get_resource_by_sync_id("notes.folder", parent_id)["parent_id"]
        is None
    )


def test_unrepresentable_local_nfkc_collision_is_reviewed_not_merged(
    repository: NotesOrganizationRepository,
) -> None:
    first, second = _id(5), _id(6)
    assert (
        _apply(
            repository,
            domain="notes.folder",
            object_id=first,
            payload={"name": "A", "parent_sync_id": None},
        ).status
        == "applied"
    )
    connection = repository.db.get_connection()
    projection_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folders ORDER BY id"
        ).fetchall()
    ]
    heads_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ]
    result = _apply(
        repository,
        domain="notes.folder",
        object_id=second,
        payload={"name": "Ａ", "parent_sync_id": None},
    )

    assert result.status == "blocked"
    assert result.reason_code == "local_representation_collision"
    assert repository.get_resource_by_sync_id("notes.folder", second) is None
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folders ORDER BY id"
        ).fetchall()
    ] == projection_before
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ] == heads_before
    first_local_id = repository.get_resource_by_sync_id("notes.folder", first)["id"]
    review = connection.execute(
        "SELECT review_id, server_profile_id, dataset_id, domain, local_object_id, "
        "remote_object_id, collision_key, display_name, portable_path, state, "
        "resolution, resolved_at FROM notes_organization_adoption_reviews"
    ).fetchone()
    expected_review_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"profile-a:dataset-a:notes.folder:{first_local_id}",
        )
    )
    assert tuple(review) == (
        expected_review_id,
        "profile-a",
        "dataset-a",
        "notes.folder",
        first_local_id,
        second,
        "ａ",
        "Ａ",
        "ａ",
        "open",
        None,
        None,
    )


@pytest.mark.parametrize("mutation", ("rename", "move"))
def test_folder_rewrite_collision_records_exact_content_free_review(
    repository: NotesOrganizationRepository,
    mutation: str,
) -> None:
    target_sync_id, external_sync_id = _id(51), _id(52)
    if mutation == "rename":
        _apply(
            repository,
            domain="notes.folder",
            object_id=target_sync_id,
            payload={"name": "Original", "parent_sync_id": None},
        )
        _apply(
            repository,
            domain="notes.folder",
            object_id=external_sync_id,
            payload={"name": "Occupied", "parent_sync_id": None},
        )
        payload = {"name": "Occupied", "parent_sync_id": None}
        expected_path = "occupied"
    else:
        source_parent, destination_parent = _id(53), _id(54)
        _apply(
            repository,
            domain="notes.folder",
            object_id=source_parent,
            payload={"name": "Source", "parent_sync_id": None},
        )
        _apply(
            repository,
            domain="notes.folder",
            object_id=destination_parent,
            payload={"name": "Destination", "parent_sync_id": None},
        )
        _apply(
            repository,
            domain="notes.folder",
            object_id=target_sync_id,
            payload={"name": "Leaf", "parent_sync_id": source_parent},
        )
        _apply(
            repository,
            domain="notes.folder",
            object_id=external_sync_id,
            payload={"name": "Leaf", "parent_sync_id": destination_parent},
        )
        payload = {"name": "Leaf", "parent_sync_id": destination_parent}
        expected_path = "destination/leaf"

    connection = repository.db.get_connection()
    external_local_id = repository.get_resource_by_sync_id(
        "notes.folder", external_sync_id
    )["id"]
    projection_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folders ORDER BY id"
        ).fetchall()
    ]
    heads_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ]

    result = _apply(
        repository,
        domain="notes.folder",
        object_id=target_sync_id,
        payload=payload,
        revision=2,
    )

    assert result.status == "blocked"
    assert result.reason_code == "local_representation_collision"
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folders ORDER BY id"
        ).fetchall()
    ] == projection_before
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ] == heads_before
    review = connection.execute(
        "SELECT review_id, server_profile_id, dataset_id, domain, local_object_id, "
        "remote_object_id, collision_key, display_name, portable_path, state, "
        "resolution, resolved_at FROM notes_organization_adoption_reviews"
    ).fetchone()
    assert tuple(review) == (
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"profile-a:dataset-a:notes.folder:{external_local_id}",
            )
        ),
        "profile-a",
        "dataset-a",
        "notes.folder",
        external_local_id,
        target_sync_id,
        expected_path,
        str(payload["name"]),
        expected_path,
        "open",
        None,
        None,
    )


def test_folder_restore_collision_records_review_and_preserves_tombstone_head(
    repository: NotesOrganizationRepository,
) -> None:
    target_sync_id, external_sync_id = _id(55), _id(56)
    payload = {"name": "Reserved", "parent_sync_id": None}
    _apply(
        repository,
        domain="notes.folder",
        object_id=target_sync_id,
        payload=payload,
    )
    _apply(
        repository,
        domain="notes.folder",
        object_id=target_sync_id,
        payload={},
        operation="tombstone",
        revision=2,
    )
    _apply(
        repository,
        domain="notes.folder",
        object_id=external_sync_id,
        payload=payload,
    )
    connection = repository.db.get_connection()
    external_local_id = repository.get_resource_by_sync_id(
        "notes.folder", external_sync_id
    )["id"]
    projection_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folders ORDER BY id"
        ).fetchall()
    ]
    heads_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ]

    result = _apply(
        repository,
        domain="notes.folder",
        object_id=target_sync_id,
        payload=payload,
        revision=3,
        restore_intent=True,
    )

    assert result.status == "blocked"
    assert result.reason_code == "local_representation_collision"
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folders ORDER BY id"
        ).fetchall()
    ] == projection_before
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ] == heads_before
    review = connection.execute(
        "SELECT review_id, server_profile_id, dataset_id, domain, local_object_id, "
        "remote_object_id, collision_key, display_name, portable_path, state, "
        "resolution, resolved_at FROM notes_organization_adoption_reviews"
    ).fetchone()
    assert tuple(review) == (
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"profile-a:dataset-a:notes.folder:{external_local_id}",
            )
        ),
        "profile-a",
        "dataset-a",
        "notes.folder",
        external_local_id,
        target_sync_id,
        "reserved",
        "Reserved",
        "reserved",
        "open",
        None,
        None,
    )


@pytest.mark.parametrize(
    ("domain", "table", "column", "deleted"),
    (
        ("notes.keyword", "keywords", "keyword", 0),
        ("notes.keyword", "keywords", "keyword", 1),
        ("notes.keyword_collection", "keyword_collections", "name", 0),
        ("notes.keyword_collection", "keyword_collections", "name", 1),
    ),
)
def test_keyword_and_collection_local_uniqueness_collisions_open_exact_review(
    repository: NotesOrganizationRepository,
    domain: str,
    table: str,
    column: str,
    deleted: int,
) -> None:
    connection = repository.db.get_connection()
    with repository.db.transaction() as cursor:
        if table == "keywords":
            cursor.execute(
                "INSERT INTO keywords(keyword, deleted, client_id, version) "
                "VALUES ('Collision', ?, 'local', 1)",
                (deleted,),
            )
        else:
            cursor.execute(
                "INSERT INTO keyword_collections(name, parent_id, deleted, client_id, version) "
                "VALUES ('Collision', NULL, ?, 'local', 1)",
                (deleted,),
            )
        local_id = cursor.lastrowid
    remote_id = _id(7 if domain == "notes.keyword" else 8)
    payload = (
        {"keyword": "collision"}
        if domain == "notes.keyword"
        else {"name": "collision", "parent_sync_id": None}
    )
    projection_before = [
        tuple(row)
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    ]
    heads_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ]

    result = _apply(
        repository,
        domain=domain,
        object_id=remote_id,
        payload=payload,
    )

    assert result.status == "blocked"
    assert result.reason_code == "local_representation_collision"
    assert [
        tuple(row)
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    ] == projection_before
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM notes_organization_heads ORDER BY domain, object_id"
        ).fetchall()
    ] == heads_before
    review = connection.execute(
        "SELECT review_id, server_profile_id, dataset_id, domain, local_object_id, "
        "remote_object_id, collision_key, display_name, portable_path, state, "
        "resolution, resolved_at FROM notes_organization_adoption_reviews"
    ).fetchone()
    expected_review_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"profile-a:dataset-a:{domain}:{local_id}",
        )
    )
    assert tuple(review) == (
        expected_review_id,
        "profile-a",
        "dataset-a",
        domain,
        str(local_id),
        remote_id,
        "collision",
        "collision",
        None,
        "open",
        None,
        None,
    )


def test_all_link_domains_verify_identity_and_materialize(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id, collection_id, folder_id, note_id = (_id(10), _id(11), _id(12), _id(13))
    conversation_id = "conversation-local-id"
    connection = repository.db.get_connection()
    repository.db.add_note("Note", "Body", note_id=note_id)
    repository.db.add_conversation(
        {"id": conversation_id, "root_id": conversation_id, "title": "Chat"}
    )
    _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload={"keyword": "sqlite"},
    )
    _apply(
        repository,
        domain="notes.keyword_collection",
        object_id=collection_id,
        payload={"name": "Database", "parent_sync_id": None},
    )
    _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={"name": "Lessons", "parent_sync_id": None},
    )

    note_link = {
        "subject_type": "note",
        "subject_id": note_id,
        "keyword_sync_id": keyword_id,
    }
    conversation_link = {
        "subject_type": "conversation",
        "subject_id": conversation_id,
        "keyword_sync_id": keyword_id,
    }
    collection_link = {
        "collection_sync_id": collection_id,
        "keyword_sync_id": keyword_id,
    }
    folder_link = {"note_id": note_id, "folder_sync_id": folder_id}
    for domain, payload, members in (
        ("notes.keyword_link", note_link, ("note", note_id, keyword_id)),
        (
            "notes.keyword_link",
            conversation_link,
            ("conversation", conversation_id, keyword_id),
        ),
        ("notes.keyword_collection_link", collection_link, (collection_id, keyword_id)),
        ("notes.folder_link", folder_link, (note_id, folder_id)),
    ):
        object_id = organization_link_id(domain, members)
        assert (
            _apply(
                repository, domain=domain, object_id=object_id, payload=payload
            ).status
            == "applied"
        )

    with pytest.raises(NotesOrganizationValidationError):
        _apply(
            repository,
            domain="notes.folder_link",
            object_id="wrong",
            payload=folder_link,
        )

    assert connection.execute("SELECT COUNT(*) FROM note_keywords").fetchone()[0] == 1
    assert (
        connection.execute("SELECT COUNT(*) FROM conversation_keywords").fetchone()[0]
        == 1
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM collection_keywords").fetchone()[0]
        == 1
    )
    assert repository.effective_folder_sync_ids(note_id) == (folder_id,)


def test_apply_uses_the_supplied_cursor_without_a_nested_transaction(
    repository: NotesOrganizationRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyword_id = _id(14)
    manager = repository.db.transaction()
    with manager as cursor:
        monkeypatch.setattr(
            repository.db,
            "transaction",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("nested transaction opened")
            ),
        )
        result = repository.apply_envelope(
            cursor,
            dataset_id="dataset-a",
            domain="notes.keyword",
            object_id=keyword_id,
            operation="upsert",
            payload={"keyword": "cursor-owned"},
            object_revision=1,
            object_hash=_hash({"keyword": "cursor-owned"}),
            server_cursor="cursor-1",
        )

    assert result.status == "applied"


def test_apply_rejects_foreign_cursor_without_writes_or_caller_rollback() -> None:
    first = CharactersRAGDB(":memory:", client_id="cursor-owner-a")
    second = CharactersRAGDB(":memory:", client_id="cursor-owner-b")
    repository = NotesOrganizationRepository(first, server_profile_id="profile-a")
    try:
        with second.transaction() as cursor:
            cursor.execute("INSERT INTO keywords(keyword) VALUES ('before-guard')")
            with pytest.raises(ValueError, match="repository-owned connection"):
                repository.apply_envelope(
                    cursor,
                    dataset_id="dataset-a",
                    domain="notes.keyword",
                    object_id=_id(15),
                    operation="upsert",
                    payload={"keyword": "foreign"},
                    object_revision=1,
                    object_hash=_hash({"keyword": "foreign"}),
                    server_cursor="cursor-1",
                )
            cursor.execute("INSERT INTO keywords(keyword) VALUES ('after-guard')")

        assert (
            first.get_connection()
            .execute("SELECT COUNT(*) FROM keywords")
            .fetchone()[0]
            == 0
        )
        assert (
            second.get_connection()
            .execute("SELECT COUNT(*) FROM keywords")
            .fetchone()[0]
            == 2
        )
        for db in (first, second):
            assert (
                db.get_connection()
                .execute("SELECT COUNT(*) FROM notes_organization_heads")
                .fetchone()[0]
                == 0
            )
    finally:
        first.close_connection()
        second.close_connection()


def test_record_intent_rejects_foreign_cursor_without_writes_or_caller_rollback() -> (
    None
):
    first = CharactersRAGDB(":memory:", client_id="intent-owner-a")
    second = CharactersRAGDB(":memory:", client_id="intent-owner-b")
    repository = NotesOrganizationRepository(first, server_profile_id="profile-a")
    try:
        with second.transaction() as cursor:
            cursor.execute(
                "INSERT INTO keywords(keyword) VALUES ('before-intent-guard')"
            )
            with pytest.raises(ValueError, match="repository-owned connection"):
                repository.record_intent(
                    cursor,
                    profile="profile-a",
                    dataset="dataset-a",
                    domain="notes.keyword",
                    object_id=_id(16),
                    operation="upsert",
                    payload={"keyword": "foreign-intent"},
                    source_version=1,
                )
            cursor.execute(
                "INSERT INTO keywords(keyword) VALUES ('after-intent-guard')"
            )

        assert (
            first.get_connection()
            .execute("SELECT COUNT(*) FROM keywords")
            .fetchone()[0]
            == 0
        )
        assert (
            second.get_connection()
            .execute("SELECT COUNT(*) FROM keywords")
            .fetchone()[0]
            == 2
        )
        for db in (first, second):
            assert (
                db.get_connection()
                .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
                .fetchone()[0]
                == 0
            )
    finally:
        first.close_connection()
        second.close_connection()


def test_folder_suppression_preserves_manual_and_managed_provenance(
    repository: NotesOrganizationRepository,
) -> None:
    folder_id, note_id = _id(20), _id(21)
    repository.db.add_note("Note", "Body", note_id=note_id)
    _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={"name": "Lessons", "parent_sync_id": None},
    )
    payload = {"note_id": note_id, "folder_sync_id": folder_id}
    link_id = organization_link_id("notes.folder_link", (note_id, folder_id))
    folder_local_id = repository.get_resource_by_sync_id("notes.folder", folder_id)[
        "id"
    ]
    with repository.db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folder_memberships(id, folder_id, note_id, ownership, owner_id, "
            "owner_active, version, deleted, created_at, modified_at) "
            "VALUES (?, ?, ?, 'managed', 'source-owner', 1, 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (_id(22), folder_local_id, note_id),
        )

    _apply(repository, domain="notes.folder_link", object_id=link_id, payload=payload)
    before = (
        repository.db.get_connection()
        .execute(
            "SELECT ownership, owner_id, deleted FROM note_folder_memberships "
            "ORDER BY ownership, owner_id"
        )
        .fetchall()
    )
    assert [(row["ownership"], row["owner_id"], row["deleted"]) for row in before] == [
        ("managed", "source-owner", 0),
        ("manual", "", 0),
    ]

    assert (
        _apply(
            repository,
            domain="notes.folder_link",
            object_id=link_id,
            payload=payload,
            operation="tombstone",
            revision=2,
        ).status
        == "applied"
    )
    assert repository.effective_folder_sync_ids(note_id) == ()
    rows = (
        repository.db.get_connection()
        .execute(
            "SELECT ownership, deleted FROM note_folder_memberships ORDER BY ownership"
        )
        .fetchall()
    )
    assert [(row["ownership"], row["deleted"]) for row in rows] == [
        ("managed", 0),
        ("manual", 1),
    ]

    assert (
        _apply(
            repository,
            domain="notes.folder_link",
            object_id=link_id,
            payload=payload,
            revision=3,
            restore_intent=True,
        ).status
        == "applied"
    )
    assert repository.effective_folder_sync_ids(note_id) == (folder_id,)


def test_folder_link_uses_device_local_membership_identity() -> None:
    folder_id, note_id = _id(44), _id(45)
    link_id = organization_link_id("notes.folder_link", (note_id, folder_id))
    membership_ids: list[str] = []
    databases: list[CharactersRAGDB] = []
    try:
        for client_id in ("device-a", "device-b"):
            db = CharactersRAGDB(":memory:", client_id=client_id)
            databases.append(db)
            repository = NotesOrganizationRepository(db, server_profile_id="profile-a")
            db.add_note("Note", "Body", note_id=note_id)
            _apply(
                repository,
                domain="notes.folder",
                object_id=folder_id,
                payload={"name": "Portable", "parent_sync_id": None},
            )
            assert (
                _apply(
                    repository,
                    domain="notes.folder_link",
                    object_id=link_id,
                    payload={"note_id": note_id, "folder_sync_id": folder_id},
                ).status
                == "applied"
            )
            membership = (
                db.get_connection()
                .execute(
                    "SELECT id FROM note_folder_memberships "
                    "WHERE ownership = 'manual' AND owner_id = ''"
                )
                .fetchone()
            )
            membership_ids.append(str(membership["id"]))

        assert len(set(membership_ids)) == 2
        assert all(uuid.UUID(item).version == 4 for item in membership_ids)
    finally:
        for db in databases:
            db.close_connection()


def test_folder_link_retries_device_local_membership_id_collision(
    repository: NotesOrganizationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder_id, note_id, other_note_id = _id(46), _id(47), _id(48)
    collision_id, allocated_id = _id(49), _id(50)
    repository.db.add_note("Target", "Body", note_id=note_id)
    repository.db.add_note("Other", "Body", note_id=other_note_id)
    _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={"name": "Portable", "parent_sync_id": None},
    )
    folder = repository.get_resource_by_sync_id("notes.folder", folder_id)
    assert folder is not None
    with repository.db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folder_memberships(id, folder_id, note_id, ownership, "
            "owner_id, owner_active, version, deleted, created_at, modified_at) "
            "VALUES (?, ?, ?, 'manual', '', 1, 1, 0, ?, ?)",
            (collision_id, folder["id"], other_note_id, "2026-01-01", "2026-01-01"),
        )

    generated_ids = iter((uuid.UUID(collision_id), uuid.UUID(allocated_id)))
    monkeypatch.setattr(uuid, "uuid4", lambda: next(generated_ids))
    link_id = organization_link_id("notes.folder_link", (note_id, folder_id))
    assert (
        _apply(
            repository,
            domain="notes.folder_link",
            object_id=link_id,
            payload={"note_id": note_id, "folder_sync_id": folder_id},
        ).status
        == "applied"
    )

    membership = (
        repository.db.get_connection()
        .execute(
            "SELECT id FROM note_folder_memberships WHERE note_id = ?",
            (note_id,),
        )
        .fetchone()
    )
    assert membership["id"] == allocated_id


def test_folder_link_membership_id_exhaustion_is_bounded_without_projection_write(
    repository: NotesOrganizationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder_id, note_id, other_note_id = _id(57), _id(58), _id(59)
    collision_id = _id(60)
    repository.db.add_note("Target", "Body", note_id=note_id)
    repository.db.add_note("Other", "Body", note_id=other_note_id)
    _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={"name": "Portable", "parent_sync_id": None},
    )
    folder = repository.get_resource_by_sync_id("notes.folder", folder_id)
    assert folder is not None
    with repository.db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folder_memberships(id, folder_id, note_id, ownership, "
            "owner_id, owner_active, version, deleted, created_at, modified_at) "
            "VALUES (?, ?, ?, 'manual', '', 1, 1, 0, ?, ?)",
            (collision_id, folder["id"], other_note_id, "2026-01-01", "2026-01-01"),
        )
        cursor.execute(
            "INSERT INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at) "
            "VALUES (?, ?, ?)",
            (note_id, folder_id, "2026-01-01"),
        )
    connection = repository.db.get_connection()
    memberships_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folder_memberships ORDER BY id"
        ).fetchall()
    ]
    suppressions_before = [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folder_sync_suppressions ORDER BY note_id, folder_sync_id"
        ).fetchall()
    ]
    attempts = 0

    def colliding_uuid() -> uuid.UUID:
        nonlocal attempts
        attempts += 1
        if attempts > organization_repository.LOCAL_ID_ALLOCATION_ATTEMPTS:
            raise AssertionError("local identity allocation did not stop")
        return uuid.UUID(collision_id)

    monkeypatch.setattr(uuid, "uuid4", colliding_uuid)
    link_id = organization_link_id("notes.folder_link", (note_id, folder_id))
    result = _apply(
        repository,
        domain="notes.folder_link",
        object_id=link_id,
        payload={"note_id": note_id, "folder_sync_id": folder_id},
    )

    assert result.status == "blocked"
    assert result.reason_code == "projection_id_exhausted"
    assert attempts == organization_repository.LOCAL_ID_ALLOCATION_ATTEMPTS
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folder_memberships ORDER BY id"
        ).fetchall()
    ] == memberships_before
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM note_folder_sync_suppressions ORDER BY note_id, folder_sync_id"
        ).fetchall()
    ] == suppressions_before


def test_folder_tombstone_hides_only_explicit_parent_and_restore_reveals_links(
    repository: NotesOrganizationRepository,
) -> None:
    parent_sync_id, child_sync_id, note_id = _id(23), _id(24), _id(25)
    repository.db.add_note("Note", "Body", note_id=note_id)
    _apply(
        repository,
        domain="notes.folder",
        object_id=parent_sync_id,
        payload={"name": "Parent", "parent_sync_id": None},
    )
    _apply(
        repository,
        domain="notes.folder",
        object_id=child_sync_id,
        payload={"name": "Child", "parent_sync_id": parent_sync_id},
    )
    link_payload = {"note_id": note_id, "folder_sync_id": child_sync_id}
    link_id = organization_link_id("notes.folder_link", (note_id, child_sync_id))
    _apply(
        repository, domain="notes.folder_link", object_id=link_id, payload=link_payload
    )
    connection = repository.db.get_connection()
    child_before = tuple(
        connection.execute(
            "SELECT * FROM note_folders WHERE sync_id = ?", (child_sync_id,)
        ).fetchone()
    )
    child_head_before = tuple(
        connection.execute(
            "SELECT * FROM notes_organization_heads "
            "WHERE domain = 'notes.folder' AND object_id = ?",
            (child_sync_id,),
        ).fetchone()
    )

    _apply(
        repository,
        domain="notes.folder",
        object_id=parent_sync_id,
        payload={},
        operation="tombstone",
        revision=2,
    )

    parent = repository.get_resource_by_sync_id("notes.folder", parent_sync_id)
    child = repository.get_resource_by_sync_id("notes.folder", child_sync_id)
    assert parent["deleted"] == 1
    assert child["deleted"] == 0
    assert tuple(child) == child_before
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM notes_organization_heads "
                "WHERE domain = 'notes.folder' AND object_id = ?",
                (child_sync_id,),
            ).fetchone()
        )
        == child_head_before
    )
    assert repository.effective_folder_sync_ids(note_id) == ()

    _apply(
        repository,
        domain="notes.folder",
        object_id=parent_sync_id,
        payload={"name": "Parent", "parent_sync_id": None},
        revision=3,
        restore_intent=True,
    )
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM note_folders WHERE sync_id = ?", (child_sync_id,)
            ).fetchone()
        )
        == child_before
    )
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM notes_organization_heads "
                "WHERE domain = 'notes.folder' AND object_id = ?",
                (child_sync_id,),
            ).fetchone()
        )
        == child_head_before
    )
    assert repository.effective_folder_sync_ids(note_id) == (child_sync_id,)


def test_folder_rewrite_overflow_blocks_before_mutating_subtree(
    repository: NotesOrganizationRepository,
) -> None:
    parent_sync_id, child_sync_id = _id(26), _id(27)
    _apply(
        repository,
        domain="notes.folder",
        object_id=parent_sync_id,
        payload={"name": "R", "parent_sync_id": None},
    )
    _apply(
        repository,
        domain="notes.folder",
        object_id=child_sync_id,
        payload={"name": "c" * 498, "parent_sync_id": parent_sync_id},
    )
    connection = repository.db.get_connection()
    before = [
        tuple(row)
        for row in connection.execute(
            "SELECT id, parent_id, name, path, normalized_path, version, deleted "
            "FROM note_folders ORDER BY id"
        ).fetchall()
    ]

    result = _apply(
        repository,
        domain="notes.folder",
        object_id=parent_sync_id,
        payload={"name": "RR", "parent_sync_id": None},
        revision=2,
    )

    assert result.status == "blocked"
    assert result.reason_code == "invalid_path"
    after = [
        tuple(row)
        for row in connection.execute(
            "SELECT id, parent_id, name, path, normalized_path, version, deleted "
            "FROM note_folders ORDER BY id"
        ).fetchall()
    ]
    assert after == before


def test_replay_stale_and_restore_semantics_update_heads_exactly(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id = _id(30)
    payload = {"keyword": "failure"}
    first = _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload=payload,
        revision=2,
    )
    duplicate = _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload=payload,
        revision=2,
    )
    stale = _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload={"keyword": "old"},
        revision=1,
    )
    deleted = _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload={},
        operation="tombstone",
        revision=3,
    )

    assert (first.status, duplicate.status, stale.status, deleted.status) == (
        "applied",
        "duplicate",
        "stale",
        "applied",
    )
    blocked_restore = _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload=payload,
        revision=4,
    )
    assert blocked_restore.status == "blocked"
    assert blocked_restore.reason_code == "restore_intent_required"
    repeated_block = _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload=payload,
        revision=4,
    )
    assert repeated_block.reason_code == "restore_intent_required"
    assert (
        _apply(
            repository,
            domain="notes.keyword",
            object_id=keyword_id,
            payload=payload,
            revision=5,
            restore_intent=True,
        ).status
        == "applied"
    )


def test_missing_dependency_tombstone_blocks_and_same_revision_retries(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id, note_id = _id(31), _id(32)
    repository.db.add_note("Note", "Body", note_id=note_id)
    payload = {
        "subject_type": "note",
        "subject_id": note_id,
        "keyword_sync_id": keyword_id,
    }
    link_id = organization_link_id("notes.keyword_link", ("note", note_id, keyword_id))

    missing = _apply(
        repository,
        domain="notes.keyword_link",
        object_id=link_id,
        payload=payload,
        operation="tombstone",
    )
    assert missing.status == "blocked"
    assert missing.reason_code == "missing_dependency"
    assert (
        _apply(
            repository,
            domain="notes.keyword",
            object_id=keyword_id,
            payload={"keyword": "retry"},
        ).status
        == "applied"
    )
    retried = _apply(
        repository,
        domain="notes.keyword_link",
        object_id=link_id,
        payload=payload,
        operation="tombstone",
    )
    assert retried.status == "applied"

    unknown_resource = _apply(
        repository,
        domain="notes.folder",
        object_id=_id(33),
        payload={},
        operation="tombstone",
    )
    assert unknown_resource.status == "blocked"
    assert unknown_resource.reason_code == "missing_dependency"

    missing_folder_id = _id(36)
    folder_payload = {"note_id": note_id, "folder_sync_id": missing_folder_id}
    folder_link_id = organization_link_id(
        "notes.folder_link", (note_id, missing_folder_id)
    )
    missing_endpoint = _apply(
        repository,
        domain="notes.folder_link",
        object_id=folder_link_id,
        payload=folder_payload,
        operation="tombstone",
    )
    assert missing_endpoint.status == "blocked"
    assert missing_endpoint.reason_code == "missing_dependency"


def test_unknown_resource_tombstone_does_not_stale_prior_upsert_history(
    repository: NotesOrganizationRepository,
) -> None:
    folder_id = _id(39)

    missing_tombstone = _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={},
        operation="tombstone",
        revision=2,
    )
    assert (
        repository.db.get_connection()
        .execute(
            "SELECT 1 FROM notes_organization_heads "
            "WHERE domain = 'notes.folder' AND object_id = ?",
            (folder_id,),
        )
        .fetchone()
        is None
    )
    prior_upsert = _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={"name": "Arrived later", "parent_sync_id": None},
        revision=1,
    )
    retried_tombstone = _apply(
        repository,
        domain="notes.folder",
        object_id=folder_id,
        payload={},
        operation="tombstone",
        revision=2,
    )

    assert missing_tombstone.status == "blocked"
    assert missing_tombstone.reason_code == "missing_dependency"
    assert prior_upsert.status == "applied"
    assert retried_tombstone.status == "applied"
    folder = repository.get_resource_by_sync_id("notes.folder", folder_id)
    assert folder is not None
    assert folder["deleted"] == 1
    head = (
        repository.db.get_connection()
        .execute(
            "SELECT operation, object_revision, apply_state "
            "FROM notes_organization_heads "
            "WHERE domain = 'notes.folder' AND object_id = ?",
            (folder_id,),
        )
        .fetchone()
    )
    assert tuple(head) == ("tombstone", 2, "applied")


def test_tombstones_allow_known_inactive_endpoints(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id, note_id = _id(37), _id(38)
    repository.db.add_note("Note", "Body", note_id=note_id)
    _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload={"keyword": "inactive"},
    )
    connection = repository.db.get_connection()
    connection.execute("UPDATE notes SET deleted = 1 WHERE id = ?", (note_id,))
    connection.execute(
        "UPDATE keywords SET deleted = 1 WHERE sync_id = ?", (keyword_id,)
    )
    payload = {
        "subject_type": "note",
        "subject_id": note_id,
        "keyword_sync_id": keyword_id,
    }
    link_id = organization_link_id("notes.keyword_link", ("note", note_id, keyword_id))

    result = _apply(
        repository,
        domain="notes.keyword_link",
        object_id=link_id,
        payload=payload,
        operation="tombstone",
    )

    assert result.status == "applied"


def test_failed_restore_preserves_tombstone_gate(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id, note_id = _id(34), _id(35)
    repository.db.add_note("Note", "Body", note_id=note_id)
    _apply(
        repository,
        domain="notes.keyword",
        object_id=keyword_id,
        payload={"keyword": "restore"},
    )
    payload = {
        "subject_type": "note",
        "subject_id": note_id,
        "keyword_sync_id": keyword_id,
    }
    link_id = organization_link_id("notes.keyword_link", ("note", note_id, keyword_id))
    _apply(repository, domain="notes.keyword_link", object_id=link_id, payload=payload)
    _apply(
        repository,
        domain="notes.keyword_link",
        object_id=link_id,
        payload=payload,
        operation="tombstone",
        revision=2,
    )
    repository.db.get_connection().execute(
        "UPDATE keywords SET deleted = 1 WHERE sync_id = ?", (keyword_id,)
    )

    failed = _apply(
        repository,
        domain="notes.keyword_link",
        object_id=link_id,
        payload=payload,
        revision=3,
        restore_intent=True,
    )
    assert failed.status == "blocked"
    assert failed.reason_code == "missing_dependency"
    head = (
        repository.db.get_connection()
        .execute(
            "SELECT operation, object_revision, apply_state FROM notes_organization_heads "
            "WHERE domain = 'notes.keyword_link' AND object_id = ?",
            (link_id,),
        )
        .fetchone()
    )
    assert tuple(head) == ("tombstone", 2, "applied")
    refused = _apply(
        repository,
        domain="notes.keyword_link",
        object_id=link_id,
        payload=payload,
        revision=3,
    )
    assert refused.reason_code == "restore_intent_required"


def test_record_intent_is_canonical_immutable_and_idempotent(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id = _id(40)
    payload = {"keyword": "Tried and failed"}
    with repository.db.transaction() as cursor:
        intent_id = repository.record_intent(
            cursor,
            profile="profile-a",
            dataset="dataset-a",
            domain="notes.keyword",
            object_id=keyword_id,
            operation="upsert",
            payload=payload,
            source_version=7,
        )
        repeated = repository.record_intent(
            cursor,
            profile="profile-a",
            dataset="dataset-a",
            domain="notes.keyword",
            object_id=keyword_id,
            operation="upsert",
            payload=dict(payload),
            source_version=7,
        )
        existing_before = tuple(
            cursor.execute(
                "SELECT * FROM notes_organization_sync_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        )
        with pytest.raises(
            NotesOrganizationRepositoryError, match="different content"
        ) as exc_info:
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=keyword_id,
                operation="upsert",
                payload={"keyword": "Different content"},
                source_version=7,
            )
        assert exc_info.value.reason_code == "immutable_intent_conflict"
        assert (
            tuple(
                cursor.execute(
                    "SELECT * FROM notes_organization_sync_intents WHERE intent_id = ?",
                    (intent_id,),
                ).fetchone()
            )
            == existing_before
        )

    assert repeated == intent_id
    row = (
        repository.db.get_connection()
        .execute(
            "SELECT intent_id, payload_json, payload_hash, dependency_refs_json, "
            "outbox_client_envelope_id FROM notes_organization_sync_intents"
        )
        .fetchone()
    )
    assert row["payload_json"] == '{"keyword":"Tried and failed"}'
    assert row["payload_hash"] == _hash(payload)
    assert row["dependency_refs_json"] == "[]"
    assert row["outbox_client_envelope_id"] is None


def test_record_intent_canonicalizes_and_immutably_binds_restore_metadata(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id = _id(61)
    payload = {"keyword": "Restored lesson"}
    with repository.db.transaction() as cursor:
        tombstone_hash = _hash({})
        cursor.execute(
            """
            INSERT INTO notes_organization_heads(
                server_profile_id, dataset_id, domain, object_id, operation,
                schema_version, encryption_policy, payload_json, payload_hash,
                object_revision, object_hash, server_cursor, deleted,
                apply_state, applied_at, updated_at
            ) VALUES (
                'profile-a', 'dataset-a', 'notes.keyword', ?, 'tombstone',
                1, 'server_trusted_v1', '{}', ?, 2, ?, '41', 1,
                'applied', 'now', 'now'
            )
            """,
            (keyword_id, tombstone_hash, tombstone_hash),
        )
        intent_id = repository.record_intent(
            cursor,
            profile="profile-a",
            dataset="dataset-a",
            domain="notes.keyword",
            object_id=keyword_id,
            operation="upsert",
            payload=payload,
            routing_metadata={"restore_intent": True},
            source_version=3,
        )
        row = cursor.execute(
            "SELECT routing_metadata_json, base_server_cursor, "
            "base_object_revision, base_object_hash "
            "FROM notes_organization_sync_intents "
            "WHERE intent_id = ?",
            (intent_id,),
        ).fetchone()
        assert tuple(row) == (
            '{"restore_intent":true}',
            "41",
            2,
            tombstone_hash,
        )

        with pytest.raises(
            NotesOrganizationRepositoryError, match="different content"
        ) as exc_info:
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=keyword_id,
                operation="upsert",
                payload=payload,
                routing_metadata={},
                source_version=3,
            )
        assert exc_info.value.reason_code == "immutable_intent_conflict"

        with pytest.raises(ValueError, match="restore_intent"):
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=keyword_id,
                operation="upsert",
                payload=payload,
                routing_metadata={"restore_intent": False},
                source_version=4,
            )

        with pytest.raises(ValueError, match="restore_intent"):
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=keyword_id,
                operation="upsert",
                payload=payload,
                routing_metadata={"restore_intent": 1},
                source_version=5,
            )


def test_record_intent_rejects_restore_without_current_tombstone_head(
    repository: NotesOrganizationRepository,
) -> None:
    with repository.db.transaction() as cursor:
        with pytest.raises(ValueError, match="tombstone"):
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=_id(62),
                operation="upsert",
                payload={"keyword": "Not deleted"},
                routing_metadata={"restore_intent": True},
                source_version=1,
            )


def test_record_intent_rejects_restore_when_acknowledged_predecessor_is_stale(
    repository: NotesOrganizationRepository,
) -> None:
    object_id = _id(65)
    active_hash = _hash({"keyword": "Remote active head"})
    with repository.db.transaction() as cursor:
        tombstone_intent_id = repository.record_intent(
            cursor,
            profile="profile-a",
            dataset="dataset-a",
            domain="notes.keyword",
            object_id=object_id,
            operation="tombstone",
            payload={},
            source_version=1,
        )
        cursor.execute(
            "UPDATE notes_organization_sync_intents SET acknowledged_at = 'now' "
            "WHERE intent_id = ?",
            (tombstone_intent_id,),
        )
        cursor.execute(
            """
            INSERT INTO notes_organization_heads(
                server_profile_id, dataset_id, domain, object_id, operation,
                schema_version, encryption_policy, payload_json, payload_hash,
                object_revision, object_hash, server_cursor, deleted,
                apply_state, applied_at, updated_at
            ) VALUES (
                'profile-a', 'dataset-a', 'notes.keyword', ?, 'upsert',
                1, 'server_trusted_v1', '{"keyword":"Remote active head"}', ?,
                2, ?, '52', 0, 'applied', 'now', 'now'
            )
            """,
            (object_id, active_hash, active_hash),
        )

        with pytest.raises(ValueError, match="tombstone"):
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=object_id,
                operation="upsert",
                payload={"keyword": "Invalid restore"},
                routing_metadata={"restore_intent": True},
                source_version=2,
            )


def test_record_intent_sequences_pending_mutations_without_binding_a_partial_base(
    repository: NotesOrganizationRepository,
) -> None:
    object_id = _id(63)
    payloads = (
        {"keyword": "Created"},
        {"keyword": "Renamed"},
        {"keyword": "Renamed again"},
        {},
        {"keyword": "Restored offline"},
    )
    operations = ("upsert", "upsert", "upsert", "tombstone", "upsert")

    with repository.db.transaction() as cursor:
        intent_ids = [
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=object_id,
                operation=operation,
                payload=payload,
                routing_metadata=(
                    {"restore_intent": True} if source_version == 5 else None
                ),
                source_version=source_version,
            )
            for source_version, (operation, payload) in enumerate(
                zip(operations, payloads, strict=True), start=1
            )
        ]
        rows = cursor.execute(
            "SELECT intent_id, intent_sequence, predecessor_intent_id, "
            "base_server_cursor, base_object_revision, base_object_hash "
            "FROM notes_organization_sync_intents WHERE object_id = ? "
            "ORDER BY intent_sequence",
            (object_id,),
        ).fetchall()

        assert [row["intent_sequence"] for row in rows] == [1, 2, 3, 4, 5]
        assert [row["predecessor_intent_id"] for row in rows] == [
            None,
            *intent_ids[:-1],
        ]
        assert [
            (
                row["base_server_cursor"],
                row["base_object_revision"],
                row["base_object_hash"],
            )
            for row in rows
        ] == [(None, None, None)] * 5
        assert (
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain="notes.keyword",
                object_id=object_id,
                operation="tombstone",
                payload={},
                source_version=4,
            )
            == intent_ids[-2]
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM notes_organization_sync_intents "
                "WHERE object_id = ?",
                (object_id,),
            ).fetchone()[0]
            == 5
        )


def test_record_intent_uses_acknowledged_head_then_latest_pending_predecessor(
    repository: NotesOrganizationRepository,
) -> None:
    object_id = _id(64)
    head_hash = _hash({"keyword": "Accepted"})
    first_payload = {"keyword": "Offline rename"}
    with repository.db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_heads(
                server_profile_id, dataset_id, domain, object_id, operation,
                schema_version, encryption_policy, payload_json, payload_hash,
                object_revision, object_hash, server_cursor, deleted,
                apply_state, applied_at, updated_at
            ) VALUES (
                'profile-a', 'dataset-a', 'notes.keyword', ?, 'upsert',
                1, 'server_trusted_v1', '{}', ?, 7, ?, '51', 0,
                'applied', 'now', 'now'
            )
            """,
            (object_id, head_hash, head_hash),
        )
        repository.record_intent(
            cursor,
            profile="profile-a",
            dataset="dataset-a",
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            payload=first_payload,
            source_version=8,
        )
        repository.record_intent(
            cursor,
            profile="profile-a",
            dataset="dataset-a",
            domain="notes.keyword",
            object_id=object_id,
            operation="tombstone",
            payload={},
            source_version=9,
        )
        rows = cursor.execute(
            "SELECT intent_id, intent_sequence, predecessor_intent_id, "
            "base_server_cursor, base_object_revision, base_object_hash "
            "FROM notes_organization_sync_intents WHERE object_id = ? "
            "ORDER BY intent_sequence",
            (object_id,),
        ).fetchall()

    assert [
        (
            row["intent_sequence"],
            row["predecessor_intent_id"],
            row["base_server_cursor"],
            row["base_object_revision"],
            row["base_object_hash"],
        )
        for row in rows
    ] == [
        (1, None, "51", 7, head_hash),
        (2, rows[0]["intent_id"], None, None, None),
    ]


def test_record_intent_uses_canonical_dependency_domains(
    repository: NotesOrganizationRepository,
) -> None:
    keyword_id, note_id, folder_id = _id(41), _id(42), _id(43)
    conversation_id = "conversation-dependency"
    cases = (
        (
            "notes.keyword_link",
            organization_link_id("notes.keyword_link", ("note", note_id, keyword_id)),
            {
                "subject_type": "note",
                "subject_id": note_id,
                "keyword_sync_id": keyword_id,
            },
            [
                {"domain": "notes.keyword", "object_id": keyword_id},
                {"domain": "notes.note", "object_id": note_id},
            ],
        ),
        (
            "notes.keyword_link",
            organization_link_id(
                "notes.keyword_link", ("conversation", conversation_id, keyword_id)
            ),
            {
                "subject_type": "conversation",
                "subject_id": conversation_id,
                "keyword_sync_id": keyword_id,
            },
            [
                {"domain": "notes.keyword", "object_id": keyword_id},
                {"domain": "chat.conversation", "object_id": conversation_id},
            ],
        ),
        (
            "notes.folder_link",
            organization_link_id("notes.folder_link", (note_id, folder_id)),
            {"note_id": note_id, "folder_sync_id": folder_id},
            [
                {"domain": "notes.note", "object_id": note_id},
                {"domain": "notes.folder", "object_id": folder_id},
            ],
        ),
    )
    with repository.db.transaction() as cursor:
        for source_version, (domain, object_id, payload, expected) in enumerate(
            cases, start=1
        ):
            repository.record_intent(
                cursor,
                profile="profile-a",
                dataset="dataset-a",
                domain=domain,
                object_id=object_id,
                operation="upsert",
                payload=payload,
                source_version=source_version,
            )
            row = cursor.execute(
                "SELECT dependency_refs_json FROM notes_organization_sync_intents "
                "WHERE domain = ? AND object_id = ?",
                (domain, object_id),
            ).fetchone()
            assert json.loads(row["dependency_refs_json"]) == expected
