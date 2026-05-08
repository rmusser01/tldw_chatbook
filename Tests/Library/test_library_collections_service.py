"""Library Collections local service and persistence contracts."""

from __future__ import annotations

import sqlite3
from itertools import count
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_collections_service import (
    DuplicateLibraryCollectionItem,
    DuplicateLibraryCollectionName,
    InvalidLibraryCollectionName,
    LocalLibraryCollectionsService,
)


def _service(tmp_path: Path) -> LocalLibraryCollectionsService:
    id_counter = count(1)
    timestamps = iter(
        (
            "2026-05-08T04:00:00Z",
            "2026-05-08T04:01:00Z",
            "2026-05-08T04:02:00Z",
            "2026-05-08T04:03:00Z",
            "2026-05-08T04:04:00Z",
        )
    )
    return LocalLibraryCollectionsService(
        LibraryCollectionsDB(tmp_path / "library_collections.db"),
        id_factory=lambda: f"collection-{next(id_counter)}",
        now_factory=lambda: next(timestamps),
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


def test_rename_collection_updates_name_description_and_updated_at(tmp_path: Path) -> None:
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


def test_schema_version_and_foreign_keys_are_initialized(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "library_collections.db")

    assert db.get_schema_version() == 1
    with db.connection() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_item_membership_allows_same_source_across_collections_only(tmp_path: Path) -> None:
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
        assert conn.execute("SELECT COUNT(*) FROM library_collections").fetchone()[0] == 0
