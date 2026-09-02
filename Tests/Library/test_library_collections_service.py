"""Library Collections local service and persistence contracts."""

from __future__ import annotations

import sqlite3
from itertools import count
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence
from tldw_chatbook.Library.library_collections_service import (
    LegacyCollectionsReadOnlyError,
    LibraryCollectionRecord,
    LibraryCollectionsServiceError,
    LocalLibraryCollectionsService,
)


EXPECTED_DEFAULT_COLLECTION_LIST_LIMIT = 200


def _service(_tmp_path: Path) -> LocalLibraryCollectionsService:
    id_counter = count(1)
    timestamp_counter = count(0)
    return LocalLibraryCollectionsService(
        LibraryCollectionsDB(":memory:"),
        id_factory=lambda: f"collection-{next(id_counter)}",
        now_factory=lambda: f"2026-05-08T04:{next(timestamp_counter):02d}:00Z",
    )


def _seed_equal_timestamp_collections(service: LocalLibraryCollectionsService) -> None:
    rows = [
        ("collection-b", "A"),
        ("collection-a", "a"),
        *((f"collection-{index:02d}", f"B {index:02d}") for index in range(1, 44)),
    ]
    with service.db.transaction() as conn:
        conn.executemany(
            """
            INSERT INTO library_collections (
                collection_id,
                name,
                description,
                created_at,
                updated_at
            )
            VALUES (?, ?, '', '2026-05-08T04:00:00Z', '2026-05-08T04:00:00Z')
            """,
            rows,
        )


def _seed_collection(
    service: LocalLibraryCollectionsService,
    name: str,
    *,
    description: str = "",
) -> LibraryCollectionRecord:
    collection_id = service._id_factory()
    now = service._now_factory()
    with service.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO library_collections (
                collection_id,
                name,
                description,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (collection_id, name, description, now, now),
        )
    collection = service.get_collection(collection_id)
    assert collection is not None
    return collection


def _seed_membership(
    service: LocalLibraryCollectionsService,
    collection_id: str,
    *,
    source_type: str,
    source_id: str,
    title: str = "",
) -> str:
    membership_id = service._id_factory()
    with service.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO library_collection_items (
                membership_id,
                collection_id,
                source_type,
                source_id,
                title,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                membership_id,
                collection_id,
                source_type.strip().lower(),
                source_id,
                title,
                service._now_factory(),
            ),
        )
    return membership_id


def _seed_delete(
    service: LocalLibraryCollectionsService,
    collection_id: str,
) -> bool:
    now = service._now_factory()
    with service.db.transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE library_collections
            SET deleted_at = ?, updated_at = ?
            WHERE collection_id = ? AND deleted_at IS NULL
            """,
            (now, now, collection_id),
        )
    return cursor.rowcount > 0


def test_list_collections_returns_empty_list_initially(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.list_collections() == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda service: service.create_collection("New"),
        lambda service: service.rename_collection("legacy-1", "Renamed"),
        lambda service: service.delete_collection("legacy-1"),
        lambda service: service.restore_collection("legacy-1"),
        lambda service: service.add_item_to_collection(
            "legacy-1",
            source_type="note",
            source_id="note-1",
        ),
    ],
)
def test_legacy_mutations_fail_before_any_database_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    service = _service(tmp_path)

    def forbidden_database_access():
        raise AssertionError("legacy mutation reached the database")

    monkeypatch.setattr(service.db, "connection", forbidden_database_access)
    monkeypatch.setattr(service.db, "transaction", forbidden_database_access)

    with pytest.raises(LegacyCollectionsReadOnlyError) as caught:
        mutate(service)

    assert caught.value.reason == "legacy_read_only"
    assert str(caught.value) == "legacy_read_only"


def test_collections_user_content_evidence_counts_only_active_local_collections(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    assert service.get_library_user_content_evidence() is LibraryContentEvidence.EMPTY

    collection = _seed_collection(service, "Private collection")
    evidence = service.get_library_user_content_evidence()
    assert type(evidence) is LibraryContentEvidence
    assert evidence is LibraryContentEvidence.HAS_USER_CONTENT

    assert _seed_delete(service, collection.collection_id)
    assert service.get_library_user_content_evidence() is LibraryContentEvidence.EMPTY


def test_schema_version_and_foreign_keys_are_initialized(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(":memory:")

    assert db.get_schema_version() == 3
    with db.connection() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_transaction_rolls_back_failed_collection_write(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(":memory:")

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO library_collections (
                    collection_id,
                    name,
                    description,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "collection-rollback",
                    "Rollback Candidate",
                    "",
                    "2026-05-08T04:00:00Z",
                    "2026-05-08T04:00:00Z",
                ),
            )
            raise RuntimeError("force rollback")

    with db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM library_collections").fetchone()[0] == 0
        )


def test_list_collections_uses_default_limit_and_accepts_explicit_limit(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    for index in range(EXPECTED_DEFAULT_COLLECTION_LIST_LIMIT + 2):
        _seed_collection(service, f"Collection {index:03d}")

    default_records = service.list_collections()
    explicit_records = service.list_collections(limit=3)

    assert len(default_records) == EXPECTED_DEFAULT_COLLECTION_LIST_LIMIT
    assert len(explicit_records) == 3
    assert [record.name for record in explicit_records] == [
        "Collection 000",
        "Collection 001",
        "Collection 002",
    ]


def test_sqlite_errors_are_normalized_to_service_errors(tmp_path: Path) -> None:
    service = _service(tmp_path)

    def broken_connection():
        raise sqlite3.OperationalError("database is locked")

    service.db.connection = broken_connection

    with pytest.raises(
        LibraryCollectionsServiceError, match="Library Collections storage failed"
    ):
        service.list_collections()


# ---------------------------------------------------------------------------
# task-1337 (plan Task 4): Library agent read seams for Collections.
# Bounded list/search pages with exact totals, honest match evidence over
# name/description/direct stored member titles, and membership pages whose
# supported source identities map through the shared public-ID codec.
# ---------------------------------------------------------------------------

SUPPORTED_MEMBER_SOURCE_TYPES = (
    "media",
    "note",
    "prompt",
    "skill",
    "conversation",
    "collection",
)


def test_list_library_collections_exact_total_and_stable_page(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = _seed_collection(service, "Alpha")
    second = _seed_collection(service, "Beta")
    third = _seed_collection(service, "Gamma")

    page = service.list_library_collections(limit=2, offset=0)

    assert page["total"] == 3
    assert page["offset"] == 0
    assert page["limit"] == 2
    assert [item["collection_id"] for item in page["items"]] == [
        first.collection_id,
        second.collection_id,
    ]
    assert page["items"][0]["name"] == "Alpha"

    rest = service.list_library_collections(limit=2, offset=2)
    assert rest["total"] == 3
    assert [item["collection_id"] for item in rest["items"]] == [third.collection_id]


def test_collection_pages_use_stable_id_after_equal_time_and_casefolded_name(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _seed_equal_timestamp_collections(service)
    statements: list[str] = []
    with service.db.connection() as conn:
        conn.set_trace_callback(statements.append)

    first = service.list_library_collections(limit=20, offset=0)
    second = service.list_library_collections(limit=20, offset=20)
    final = service.list_library_collections(limit=20, offset=40)

    assert first["total"] == second["total"] == final["total"] == 45
    assert [len(first["items"]), len(second["items"]), len(final["items"])] == [
        20,
        20,
        5,
    ]
    assert [item["collection_id"] for item in first["items"][:3]] == [
        "collection-a",
        "collection-b",
        "collection-01",
    ]
    assert [record.collection_id for record in service.list_collections(limit=3)] == [
        "collection-a",
        "collection-b",
        "collection-01",
    ]
    with service.db.connection() as conn:
        conn.set_trace_callback(None)
    ordered_selects = [
        " ".join(statement.lower().split())
        for statement in statements
        if "from library_collections as collection" in statement.lower()
        and "order by" in statement.lower()
    ]
    assert len(ordered_selects) == 4
    assert all(
        "order by collection.created_at asc, "
        "collection.name collate nocase asc, collection.collection_id asc"
        in statement
        for statement in ordered_selects
    )


@pytest.mark.parametrize(
    ("target_id", "expected_page", "expected_rank", "expected_index"),
    [
        ("collection-a", 1, 0, 0),
        ("collection-19", 2, 20, 0),
        ("collection-43", 3, 44, 4),
    ],
)
def test_locate_library_collection_page_returns_rank_derived_owning_page(
    tmp_path: Path,
    target_id: str,
    expected_page: int,
    expected_rank: int,
    expected_index: int,
) -> None:
    service = _service(tmp_path)
    _seed_equal_timestamp_collections(service)

    located = service.locate_library_collection_page(target_id, limit=20)

    assert located is not None
    assert located["target_id"] == target_id
    assert located["target_rank"] == expected_rank
    assert located["target_index"] == expected_index
    assert located["page"] == expected_page
    assert located["offset"] == (expected_page - 1) * 20
    assert located["limit"] == 20
    assert located["total"] == 45
    assert located["items"][expected_index]["collection_id"] == target_id
    assert len(located["items"]) == (20 if expected_page < 3 else 5)


def test_locate_library_collection_page_returns_none_for_missing_or_deleted_id(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    keep = _seed_collection(service, "Keep")
    deleted = _seed_collection(service, "Deleted")
    assert _seed_delete(service, deleted.collection_id)

    assert service.locate_library_collection_page("collection-missing") is None
    assert service.locate_library_collection_page(deleted.collection_id) is None
    assert service.locate_library_collection_page(keep.collection_id) is not None


@pytest.mark.parametrize("limit", [True, "20", 0, -1, 501])
def test_collection_page_reads_reject_invalid_limits(
    tmp_path: Path, limit
) -> None:
    service = _service(tmp_path)

    with pytest.raises(LibraryCollectionsServiceError, match="limit"):
        service.list_library_collections(limit=limit)
    with pytest.raises(LibraryCollectionsServiceError, match="limit"):
        service.locate_library_collection_page("collection-1", limit=limit)


@pytest.mark.parametrize("offset", [True, "20", -1, 2**63])
def test_collection_page_reads_reject_invalid_offsets(
    tmp_path: Path, offset
) -> None:
    service = _service(tmp_path)

    with pytest.raises(LibraryCollectionsServiceError, match="offset"):
        service.list_library_collections(offset=offset)


@pytest.mark.parametrize("collection_id", ["", " ", True, 1])
def test_collection_locator_rejects_invalid_stable_ids(
    tmp_path: Path, collection_id
) -> None:
    service = _service(tmp_path)

    with pytest.raises(LibraryCollectionsServiceError, match="collection_id"):
        service.locate_library_collection_page(collection_id)


def test_list_library_collections_reports_item_counts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Counted", description="with members")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="media",
        source_id="m-1",
        title="One",
    )
    _seed_membership(
        service,
        collection.collection_id,
        source_type="note",
        source_id="n-1",
        title="Two",
    )

    page = service.list_library_collections()

    item = page["items"][0]
    assert item["item_count"] == 2
    assert item["description"] == "with members"
    assert set(item) == {
        "collection_id",
        "name",
        "description",
        "item_count",
        "created_at",
        "updated_at",
    }


def test_list_library_collections_excludes_deleted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    keep = _seed_collection(service, "Keep")
    drop = _seed_collection(service, "Drop")
    assert _seed_delete(service, drop.collection_id) is True

    page = service.list_library_collections()

    assert page["total"] == 1
    assert [item["collection_id"] for item in page["items"]] == [keep.collection_id]


def test_search_library_collections_match_branches(tmp_path: Path) -> None:
    service = _service(tmp_path)
    by_name = _seed_collection(service, "Research shelf")
    by_description = _seed_collection(
        service, "Shelf two", description="research backlog"
    )
    by_member = _seed_collection(service, "Shelf three")
    _seed_membership(
        service,
        by_member.collection_id,
        source_type="media",
        source_id="m-1",
        title="research notes",
    )

    page = service.search_library_collections(query="research")

    assert page["total"] == 3
    fields = {item["collection_id"]: item["matched_fields"] for item in page["items"]}
    assert fields[by_name.collection_id] == ["name"]
    assert fields[by_description.collection_id] == ["description"]
    assert fields[by_member.collection_id] == ["member_title"]


def test_search_library_collections_counts_multi_member_match_once(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Dedup")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="media",
        source_id="m-1",
        title="needle one",
    )
    _seed_membership(
        service,
        collection.collection_id,
        source_type="media",
        source_id="m-2",
        title="needle two",
    )

    page = service.search_library_collections(query="needle")

    assert page["total"] == 1
    assert page["items"][0]["collection_id"] == collection.collection_id
    assert page["items"][0]["matched_fields"] == ["member_title"]


def test_search_library_collections_does_not_inspect_member_content(
    tmp_path: Path,
) -> None:
    """Only the stored member *title* participates; the backing source
    identity (and never the member's content) must not be searchable."""
    service = _service(tmp_path)
    collection = _seed_collection(service, "Opaque")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="media",
        source_id="needle-raw-identity",
        title="unrelated title",
    )

    page = service.search_library_collections(query="needle")

    assert page["total"] == 0


def test_search_library_collections_exact_name_ranks_first(tmp_path: Path) -> None:
    service = _service(tmp_path)
    partial = _seed_collection(service, "needle extras")
    exact = _seed_collection(service, "needle")

    page = service.search_library_collections(query="needle")

    assert [item["collection_id"] for item in page["items"]] == [
        exact.collection_id,
        partial.collection_id,
    ]


def test_search_library_collections_like_wildcards_match_literally(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    percent = _seed_collection(service, "100% coverage")
    _seed_collection(service, "1000 coverage")

    page = service.search_library_collections(query="100%")

    assert [item["collection_id"] for item in page["items"]] == [percent.collection_id]


def test_search_library_collections_excludes_deleted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _seed_collection(service, "needle keep")
    drop = _seed_collection(service, "needle drop")
    assert _seed_delete(service, drop.collection_id) is True

    page = service.search_library_collections(query="needle")

    assert page["total"] == 1


def test_get_library_collection_missing_returns_none(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.get_library_collection("collection-missing") is None


def test_get_library_collection_pages_members_with_exact_total(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Members", description="d")
    membership_ids = [
        _seed_membership(
            service,
            collection.collection_id,
            source_type="media",
            source_id=f"m-{index}",
            title=f"Title {index}",
        )
        for index in range(3)
    ]

    first_page = service.get_library_collection(
        collection.collection_id, limit=2, offset=0
    )

    assert first_page["collection_id"] == collection.collection_id
    assert first_page["name"] == "Members"
    assert first_page["description"] == "d"
    assert first_page["member_total"] == 3
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["has_more"] is True
    assert [m["membership_id"] for m in first_page["members"]] == membership_ids[:2]
    assert all(m["title_truncated"] is False for m in first_page["members"])

    second_page = service.get_library_collection(
        collection.collection_id, limit=2, offset=2
    )
    assert second_page["has_more"] is False
    assert [m["membership_id"] for m in second_page["members"]] == membership_ids[2:]


def test_get_library_collection_supported_types_round_trip_public_ids(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    service = _service(tmp_path)
    collection = _seed_collection(service, "Typed")
    for source_type in SUPPORTED_MEMBER_SOURCE_TYPES:
        _seed_membership(
            service,
            collection.collection_id,
            source_type=source_type,
            source_id=f"{source_type}-id-1",
            title=f"{source_type} member",
        )

    detail = service.get_library_collection(collection.collection_id)

    members = {member["source_type"]: member for member in detail["members"]}
    for source_type in SUPPORTED_MEMBER_SOURCE_TYPES:
        member = members[source_type]
        assert member["source_ref"] is None
        parsed_type, parsed_raw = parse_public_id(
            member["item_id"], expected_type=source_type
        )
        assert parsed_type == source_type
        assert parsed_raw == f"{source_type}-id-1"


def test_get_library_collection_normalizes_source_type_case(tmp_path: Path) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Case")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="Media",
        source_id="m-1",
        title="t",
    )

    member = service.get_library_collection(collection.collection_id)["members"][0]

    assert member["item_id"].startswith("media:")


def test_get_library_collection_unsupported_type_gets_opaque_ref(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Refs")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="server-doc",
        source_id="doc-9",
        title="t",
    )

    member = service.get_library_collection(collection.collection_id)["members"][0]

    assert member["item_id"] is None
    assert isinstance(member["source_ref"], str)
    assert member["source_ref"]
    assert "doc-9" not in member["source_ref"]
    assert "server-doc" not in member["source_ref"]


def test_get_library_collection_bounds_member_titles(tmp_path: Path) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Titles")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="media",
        source_id="m-1",
        title="t" * 300,
    )

    member = service.get_library_collection(collection.collection_id)["members"][0]

    assert member["title_truncated"] is True
    assert len(member["title"].encode("utf-8")) <= 160


def test_get_library_collection_members_expose_no_content_fields(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = _seed_collection(service, "Lean")
    _seed_membership(
        service,
        collection.collection_id,
        source_type="note",
        source_id="n-1",
        title="t",
    )

    member = service.get_library_collection(collection.collection_id)["members"][0]

    assert set(member) == {
        "membership_id",
        "source_type",
        "item_id",
        "source_ref",
        "title",
        "title_truncated",
    }
