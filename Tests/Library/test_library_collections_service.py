"""Library Collections local service and persistence contracts."""

from __future__ import annotations

import sqlite3
from itertools import count
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence
from tldw_chatbook.Library.library_collections_service import (
    DuplicateLibraryCollectionItem,
    DuplicateLibraryCollectionName,
    InvalidLibraryCollectionDescription,
    InvalidLibraryCollectionName,
    LibraryCollectionsServiceError,
    LocalLibraryCollectionsService,
)


EXPECTED_DEFAULT_COLLECTION_LIST_LIMIT = 200


def _service(tmp_path: Path) -> LocalLibraryCollectionsService:
    id_counter = count(1)
    timestamp_counter = count(0)
    return LocalLibraryCollectionsService(
        LibraryCollectionsDB(tmp_path / "library_collections.db"),
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


def test_list_collections_returns_empty_list_initially(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.list_collections() == ()


def test_create_collection_persists_local_only_record(tmp_path: Path) -> None:
    service = _service(tmp_path)

    collection = service.create_collection(" Research ", description="Policy sources")

    assert collection.collection_id == "collection-1"
    assert collection.name == "Research"
    assert collection.description == "Policy sources"
    assert collection.item_count == 0
    assert collection.source_authority == "local"
    assert collection.sync_status == "local-only"
    assert collection.created_at == "2026-05-08T04:00:00Z"
    assert collection.updated_at == "2026-05-08T04:00:00Z"
    assert service.list_collections() == (collection,)
    assert service.get_collection("collection-1") == collection


def test_duplicate_normalized_names_are_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_collection("Research")

    with pytest.raises(DuplicateLibraryCollectionName):
        service.create_collection(" research ")


def test_rename_collection_updates_name_description_and_updated_at(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Research", description="Initial")

    renamed = service.rename_collection(
        collection.collection_id,
        "Briefing Queue",
        description="Updated",
    )

    assert renamed.collection_id == collection.collection_id
    assert renamed.name == "Briefing Queue"
    assert renamed.description == "Updated"
    assert renamed.created_at == "2026-05-08T04:00:00Z"
    assert renamed.updated_at == "2026-05-08T04:01:00Z"
    assert service.get_collection(collection.collection_id) == renamed


def test_delete_collection_hides_record_from_list_and_get(tmp_path: Path) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Research")

    assert service.delete_collection(collection.collection_id) is True

    assert service.list_collections() == ()
    assert service.get_collection(collection.collection_id) is None


def test_collections_user_content_evidence_counts_only_active_local_collections(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    assert service.get_library_user_content_evidence() is LibraryContentEvidence.EMPTY

    collection = service.create_collection("Private collection")
    evidence = service.get_library_user_content_evidence()
    assert type(evidence) is LibraryContentEvidence
    assert evidence is LibraryContentEvidence.HAS_USER_CONTENT

    assert service.delete_collection(collection.collection_id)
    assert service.get_library_user_content_evidence() is LibraryContentEvidence.EMPTY


def test_restore_collection_revives_record_with_membership(tmp_path: Path) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Research")
    service.add_item_to_collection(
        collection.collection_id,
        source_type="note",
        source_id="note-1",
        title="Evidence",
    )
    assert service.delete_collection(collection.collection_id) is True

    restored = service.restore_collection(collection.collection_id)

    assert restored.collection_id == collection.collection_id
    assert restored.name == "Research"
    assert restored.item_count == 1
    assert service.list_collections() == (restored,)


def test_schema_version_and_foreign_keys_are_initialized(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "library_collections.db")

    assert db.get_schema_version() == 1
    with db.connection() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_transaction_rolls_back_failed_collection_write(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "library_collections.db")

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


def test_item_membership_allows_same_source_across_collections_only(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    first = service.create_collection("Research")
    second = service.create_collection("Briefing")

    first_membership = service.add_item_to_collection(
        first.collection_id,
        source_type="media",
        source_id="item-1",
        title="Saved article",
    )
    second_membership = service.add_item_to_collection(
        second.collection_id,
        source_type="media",
        source_id="item-1",
        title="Saved article",
    )

    assert first_membership != second_membership
    assert service.get_collection(first.collection_id).item_count == 1
    assert service.get_collection(second.collection_id).item_count == 1
    with pytest.raises(DuplicateLibraryCollectionItem):
        service.add_item_to_collection(
            first.collection_id,
            source_type="media",
            source_id="item-1",
            title="Saved article",
        )


def test_invalid_names_are_rejected_before_sql(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(InvalidLibraryCollectionName):
        service.create_collection(" ")
    with pytest.raises(InvalidLibraryCollectionName):
        service.create_collection("<script>alert(1)</script>")
    with pytest.raises(InvalidLibraryCollectionName):
        service.create_collection("x" * 121)

    with sqlite3.connect(tmp_path / "library_collections.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM library_collections").fetchone()[0] == 0
        )


def test_descriptions_reject_unsafe_html_before_persistence(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(InvalidLibraryCollectionDescription):
        service.create_collection("Research", description="<script>alert(1)</script>")

    with sqlite3.connect(tmp_path / "library_collections.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM library_collections").fetchone()[0] == 0
        )


def test_list_collections_uses_default_limit_and_accepts_explicit_limit(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    for index in range(EXPECTED_DEFAULT_COLLECTION_LIST_LIMIT + 2):
        service.create_collection(f"Collection {index:03d}")

    default_records = service.list_collections()
    explicit_records = service.list_collections(limit=3)

    assert len(default_records) == EXPECTED_DEFAULT_COLLECTION_LIST_LIMIT
    assert len(explicit_records) == 3
    assert [record.name for record in explicit_records] == [
        "Collection 000",
        "Collection 001",
        "Collection 002",
    ]


def test_deleted_collection_name_fails_before_late_sql_integrity_error(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Research")
    assert service.delete_collection(collection.collection_id) is True

    with pytest.raises(DuplicateLibraryCollectionName, match="deleted Collection"):
        service.create_collection("research")


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
    first = service.create_collection("Alpha")
    second = service.create_collection("Beta")
    third = service.create_collection("Gamma")

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
    keep = service.create_collection("Keep")
    deleted = service.create_collection("Deleted")
    assert service.delete_collection(deleted.collection_id)

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
    collection = service.create_collection("Counted", description="with members")
    service.add_item_to_collection(
        collection.collection_id, source_type="media", source_id="m-1", title="One"
    )
    service.add_item_to_collection(
        collection.collection_id, source_type="note", source_id="n-1", title="Two"
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
    keep = service.create_collection("Keep")
    drop = service.create_collection("Drop")
    assert service.delete_collection(drop.collection_id) is True

    page = service.list_library_collections()

    assert page["total"] == 1
    assert [item["collection_id"] for item in page["items"]] == [keep.collection_id]


def test_search_library_collections_match_branches(tmp_path: Path) -> None:
    service = _service(tmp_path)
    by_name = service.create_collection("Research shelf")
    by_description = service.create_collection(
        "Shelf two", description="research backlog"
    )
    by_member = service.create_collection("Shelf three")
    service.add_item_to_collection(
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
    collection = service.create_collection("Dedup")
    service.add_item_to_collection(
        collection.collection_id,
        source_type="media",
        source_id="m-1",
        title="needle one",
    )
    service.add_item_to_collection(
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
    collection = service.create_collection("Opaque")
    service.add_item_to_collection(
        collection.collection_id,
        source_type="media",
        source_id="needle-raw-identity",
        title="unrelated title",
    )

    page = service.search_library_collections(query="needle")

    assert page["total"] == 0


def test_search_library_collections_exact_name_ranks_first(tmp_path: Path) -> None:
    service = _service(tmp_path)
    partial = service.create_collection("needle extras")
    exact = service.create_collection("needle")

    page = service.search_library_collections(query="needle")

    assert [item["collection_id"] for item in page["items"]] == [
        exact.collection_id,
        partial.collection_id,
    ]


def test_search_library_collections_like_wildcards_match_literally(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    percent = service.create_collection("100% coverage")
    service.create_collection("1000 coverage")

    page = service.search_library_collections(query="100%")

    assert [item["collection_id"] for item in page["items"]] == [percent.collection_id]


def test_search_library_collections_excludes_deleted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_collection("needle keep")
    drop = service.create_collection("needle drop")
    assert service.delete_collection(drop.collection_id) is True

    page = service.search_library_collections(query="needle")

    assert page["total"] == 1


def test_get_library_collection_missing_returns_none(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.get_library_collection("collection-missing") is None


def test_get_library_collection_pages_members_with_exact_total(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Members", description="d")
    membership_ids = [
        service.add_item_to_collection(
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
    collection = service.create_collection("Typed")
    for source_type in SUPPORTED_MEMBER_SOURCE_TYPES:
        service.add_item_to_collection(
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
    collection = service.create_collection("Case")
    service.add_item_to_collection(
        collection.collection_id, source_type="Media", source_id="m-1", title="t"
    )

    member = service.get_library_collection(collection.collection_id)["members"][0]

    assert member["item_id"].startswith("media:")


def test_get_library_collection_unsupported_type_gets_opaque_ref(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Refs")
    service.add_item_to_collection(
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
    collection = service.create_collection("Titles")
    service.add_item_to_collection(
        collection.collection_id, source_type="media", source_id="m-1", title="t" * 300
    )

    member = service.get_library_collection(collection.collection_id)["members"][0]

    assert member["title_truncated"] is True
    assert len(member["title"].encode("utf-8")) <= 160


def test_get_library_collection_members_expose_no_content_fields(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    collection = service.create_collection("Lean")
    service.add_item_to_collection(
        collection.collection_id, source_type="note", source_id="n-1", title="t"
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
