"""Crash and adoption coverage for the legacy Notes organization inventory."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
    NotesOrganizationRepositoryError,
)
from tldw_chatbook.Sync_Interop.notes_organization import organization_link_id


PROFILE = "profile-a"
DATASET = "dataset-a"
NOW = "2026-08-29T12:00:00.000Z"


def _id(number: int) -> str:
    return str(uuid.UUID(f"00000000-0000-4000-8000-{number:012d}"))


IDS = {
    "keyword_keep": _id(1),
    "keyword_deleted": _id(2),
    "keyword_remote_same_name": _id(3),
    "collection_local": _id(4),
    "collection_remote": _id(5),
    "collection_child": _id(6),
    "folder_parent": _id(7),
    "folder_child": _id(8),
    "note_enrolled": _id(9),
    "note_missing": _id(10),
}


def _inventory_types():
    from tldw_chatbook.Sync_Interop.notes_organization_inventory import (
        LegacyNotesOrganizationInventory,
    )

    return LegacyNotesOrganizationInventory


def _insert_review(
    cursor,
    *,
    review_id: str,
    domain: str,
    local_object_id: str,
    remote_object_id: str,
    resolution: str | None,
) -> None:
    state = "resolved" if resolution is not None else "open"
    cursor.execute(
        """
        INSERT INTO notes_organization_adoption_reviews(
            review_id, server_profile_id, dataset_id, domain, local_object_id,
            remote_object_id, collision_key, display_name, portable_path,
            state, resolution, created_at, updated_at, resolved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            PROFILE,
            DATASET,
            domain,
            local_object_id,
            remote_object_id,
            f"collision-{review_id}",
            f"display-{review_id}",
            state,
            resolution,
            NOW,
            NOW,
            NOW if resolution is not None else None,
        ),
    )


def _seed_inventory_db(
    path: Path,
    *,
    local_state: str = "adoption_review",
    bootstrap_id: str | None = "bootstrap-a",
    error_code: str | None = None,
) -> None:
    db = CharactersRAGDB(path, client_id="inventory-tests")
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                bootstrap_id, captured_count, expected_count, error_code,
                pull_cursor, inventory_phase, updated_at
            ) VALUES (?, ?, ?, 'ready', ?, 8, 8, ?, 'pull-cursor-a',
                      'not_started', ?)
            """,
            (PROFILE, DATASET, local_state, bootstrap_id, error_code, NOW),
        )
        cursor.execute(
            "INSERT INTO keywords(keyword, deleted, client_id, version, sync_id) "
            "VALUES ('Keep local', 0, 'legacy', 3, ?)",
            (IDS["keyword_keep"],),
        )
        keyword_keep_id = int(cursor.lastrowid)
        cursor.execute(
            "INSERT INTO keywords(keyword, deleted, client_id, version, sync_id) "
            "VALUES ('Deleted keyword', 1, 'legacy', 4, ?)",
            (IDS["keyword_deleted"],),
        )
        keyword_deleted_id = int(cursor.lastrowid)

        cursor.execute(
            "INSERT INTO keyword_collections(name, parent_id, deleted, client_id, "
            "version, sync_id) VALUES ('Deleted collection', NULL, 1, 'legacy', 5, ?)",
            (IDS["collection_local"],),
        )
        collection_parent_id = int(cursor.lastrowid)
        cursor.execute(
            "INSERT INTO keyword_collections(name, parent_id, deleted, client_id, "
            "version, sync_id) VALUES ('Collection child', ?, 0, 'legacy', 2, ?)",
            (collection_parent_id, IDS["collection_child"]),
        )
        collection_child_id = int(cursor.lastrowid)

        cursor.execute(
            """
            INSERT INTO note_folders(
                id, parent_id, name, normalized_name, path, normalized_path,
                version, deleted, created_at, modified_at, sync_id
            ) VALUES ('folder-parent-local', NULL, 'Deleted folder',
                      'deleted folder', '/Deleted folder', '/deleted folder',
                      4, 1, ?, ?, ?)
            """,
            (NOW, NOW, IDS["folder_parent"]),
        )
        cursor.execute(
            """
            INSERT INTO note_folders(
                id, parent_id, name, normalized_name, path, normalized_path,
                version, deleted, created_at, modified_at, sync_id
            ) VALUES ('folder-child-local', 'folder-parent-local', 'Folder child',
                      'folder child', '/Deleted folder/Folder child',
                      '/deleted folder/folder child', 2, 0, ?, ?, ?)
            """,
            (NOW, NOW, IDS["folder_child"]),
        )

        for note_id, title in (
            (IDS["note_enrolled"], "Enrolled note"),
            (IDS["note_missing"], "Missing dependency note"),
        ):
            cursor.execute(
                "INSERT INTO notes(id, title, content, client_id, version) "
                "VALUES (?, ?, 'private note body', 'legacy', 1)",
                (note_id, title),
            )
        for conversation_id in ("conversation-enrolled", "conversation-missing"):
            cursor.execute(
                "INSERT INTO conversations(id, root_id, title, client_id, version) "
                "VALUES (?, ?, 'Conversation', 'legacy', 1)",
                (conversation_id, conversation_id),
            )

        cursor.execute(
            "INSERT INTO note_keywords(note_id, keyword_id, created_at) VALUES (?, ?, ?)",
            (IDS["note_enrolled"], keyword_keep_id, NOW),
        )
        cursor.execute(
            "INSERT INTO note_keywords(note_id, keyword_id, created_at) VALUES (?, ?, ?)",
            (IDS["note_missing"], keyword_deleted_id, NOW),
        )
        cursor.execute(
            "INSERT INTO conversation_keywords(conversation_id, keyword_id, created_at) "
            "VALUES ('conversation-enrolled', ?, ?)",
            (keyword_deleted_id, NOW),
        )
        cursor.execute(
            "INSERT INTO conversation_keywords(conversation_id, keyword_id, created_at) "
            "VALUES ('conversation-missing', ?, ?)",
            (keyword_deleted_id, NOW),
        )
        cursor.execute(
            "INSERT INTO collection_keywords(collection_id, keyword_id, created_at) "
            "VALUES (?, ?, ?)",
            (collection_child_id, keyword_deleted_id, NOW),
        )
        cursor.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id, owner_active,
                version, deleted, created_at, modified_at
            ) VALUES ('membership-dormant', 'folder-child-local', ?, 'managed',
                      'source-a', 0, 2, 0, ?, ?)
            """,
            (IDS["note_enrolled"], NOW, NOW),
        )
        cursor.execute(
            "INSERT INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at) "
            "VALUES (?, ?, ?)",
            (IDS["note_enrolled"], IDS["folder_child"], NOW),
        )
        cursor.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id, owner_active,
                version, deleted, created_at, modified_at
            ) VALUES ('membership-deleted', 'folder-parent-local', ?, 'manual',
                      '', 1, 3, 1, ?, ?)
            """,
            (IDS["note_enrolled"], NOW, NOW),
        )
        cursor.execute(
            """
            INSERT INTO sync_log(
                entity, entity_id, operation, timestamp, client_id, version, payload
            ) VALUES ('note_keywords', 'deleted-note-keyword', 'delete', ?, 'legacy', 1, ?)
            """,
            (
                NOW,
                json.dumps(
                    {
                        "note_id": IDS["note_enrolled"],
                        "keyword_id": keyword_deleted_id,
                    }
                ),
            ),
        )
        cursor.execute(
            """
            INSERT INTO sync_log(
                entity, entity_id, operation, timestamp, client_id, version, payload
            ) VALUES ('collection_keywords', 'deleted-collection-keyword', 'delete',
                      ?, 'legacy', 1, ?)
            """,
            (
                NOW,
                json.dumps(
                    {
                        "collection_id": collection_parent_id,
                        "keyword_id": keyword_deleted_id,
                    }
                ),
            ),
        )

        _insert_review(
            cursor,
            review_id="keep-local-review",
            domain="notes.keyword",
            local_object_id=str(keyword_keep_id),
            remote_object_id=IDS["keyword_remote_same_name"],
            resolution="keep_local",
        )
        _insert_review(
            cursor,
            review_id="merge-review",
            domain="notes.keyword_collection",
            local_object_id=str(collection_parent_id),
            remote_object_id=IDS["collection_remote"],
            resolution="merge",
        )
    db.close_connection()


def _run_inventory(
    path: Path,
    *,
    after_commit: Callable[[str, str | None], None] | None = None,
    enrolled_note_ids: set[str] | None = None,
    enrolled_conversation_ids: set[str] | None = None,
):
    Inventory = _inventory_types()
    db = CharactersRAGDB(path, client_id="inventory-tests")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    runner = Inventory(
        repository,
        dataset_id=DATASET,
        enrolled_note_ids=(
            {IDS["note_enrolled"]}
            if enrolled_note_ids is None
            else enrolled_note_ids
        ),
        enrolled_conversation_ids=(
            {"conversation-enrolled"}
            if enrolled_conversation_ids is None
            else enrolled_conversation_ids
        ),
    )
    try:
        return runner.run(after_commit=after_commit)
    finally:
        db.close_connection()


def _intent_rows(path: Path) -> list[dict[str, object]]:
    db = CharactersRAGDB(path, client_id="inventory-tests")
    try:
        rows = db.get_connection().execute(
            "SELECT rowid, intent_id, domain, object_id, operation, payload_json, "
            "source_version FROM notes_organization_sync_intents ORDER BY rowid"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close_connection()


def _durable_inventory_bytes(path: Path) -> bytes:
    db = CharactersRAGDB(path, client_id="inventory-tests")
    try:
        connection = db.get_connection()
        checkpoint = connection.execute(
            "SELECT * FROM notes_organization_sync_checkpoints "
            "WHERE server_profile_id = ? AND dataset_id = ?",
            (PROFILE, DATASET),
        ).fetchone()
        intents = connection.execute(
            "SELECT * FROM notes_organization_sync_intents ORDER BY rowid"
        ).fetchall()
        return json.dumps(
            {
                "checkpoint": dict(checkpoint),
                "intents": [dict(row) for row in intents],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    finally:
        db.close_connection()


def test_inventory_module_exists_before_behavior_tests() -> None:
    assert (
        importlib.util.find_spec(
            "tldw_chatbook.Sync_Interop.notes_organization_inventory"
        )
        is not None
    )


def test_inventory_holds_until_pull_and_adoption_prerequisites(tmp_path: Path) -> None:
    pulling_path = tmp_path / "pulling.sqlite"
    _seed_inventory_db(pulling_path, local_state="pulling")
    result = _run_inventory(pulling_path)
    assert (result.status, result.reason_code) == (
        "held",
        "bootstrap_pull_incomplete",
    )
    assert _intent_rows(pulling_path) == []

    review_path = tmp_path / "review.sqlite"
    _seed_inventory_db(review_path)
    db = CharactersRAGDB(review_path, client_id="inventory-tests")
    with db.transaction() as cursor:
        _insert_review(
            cursor,
            review_id="open-review",
            domain="notes.folder",
            local_object_id="folder-child-local",
            remote_object_id=_id(30),
            resolution=None,
        )
    db.close_connection()
    result = _run_inventory(review_path)
    assert (result.status, result.reason_code) == (
        "held",
        "adoption_review_required",
    )
    assert _intent_rows(review_path) == []

    for name, options in (
        ("missing-bootstrap", {"bootstrap_id": None}),
        ("persisted-error", {"error_code": "bootstrap_failed"}),
    ):
        path = tmp_path / f"{name}.sqlite"
        _seed_inventory_db(path, **options)
        result = _run_inventory(path)
        assert (result.status, result.reason_code) == (
            "held",
            "bootstrap_pull_incomplete",
        )
        assert _intent_rows(path) == []


def test_inventory_orders_adopted_resources_links_and_evidenced_tombstones(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inventory.sqlite"
    _seed_inventory_db(path)

    result = _run_inventory(path)

    assert result.status == "complete"
    assert result.reason_code is None
    assert result.skipped_dependencies == (
        ("notes.keyword_link", "chat.conversation", "conversation-missing"),
        ("notes.keyword_link", "notes.note", IDS["note_missing"]),
    )
    rows = _intent_rows(path)
    logical = [(r["domain"], r["object_id"], r["operation"]) for r in rows]
    assert len(logical) == len(set(logical)) == 18

    resource_upserts = {
        ("notes.keyword", IDS["keyword_deleted"], "upsert"),
        ("notes.keyword_collection", IDS["collection_remote"], "upsert"),
        ("notes.keyword_collection", IDS["collection_child"], "upsert"),
        ("notes.folder", IDS["folder_parent"], "upsert"),
        ("notes.folder", IDS["folder_child"], "upsert"),
    }
    assert resource_upserts <= set(logical)
    assert not any(
        domain == "notes.keyword" and object_id == IDS["keyword_keep"]
        for domain, object_id, _operation in logical
    )
    assert logical.index(
        ("notes.keyword_collection", IDS["collection_remote"], "upsert")
    ) < logical.index(
        ("notes.keyword_collection", IDS["collection_child"], "upsert")
    )
    assert logical.index(
        ("notes.folder", IDS["folder_parent"], "upsert")
    ) < logical.index(("notes.folder", IDS["folder_child"], "upsert"))

    payloads = {
        (str(row["domain"]), str(row["operation"]), str(row["object_id"])): json.loads(
            str(row["payload_json"])
        )
        for row in rows
    }
    expected_links = {
        (
            "notes.keyword_link",
            organization_link_id(
                "notes.keyword_link",
                ("note", IDS["note_enrolled"], IDS["keyword_deleted"]),
            ),
        ): {
            "subject_type": "note",
            "subject_id": IDS["note_enrolled"],
            "keyword_sync_id": IDS["keyword_deleted"],
        },
        (
            "notes.keyword_link",
            organization_link_id(
                "notes.keyword_link",
                ("conversation", "conversation-enrolled", IDS["keyword_deleted"]),
            ),
        ): {
            "subject_type": "conversation",
            "subject_id": "conversation-enrolled",
            "keyword_sync_id": IDS["keyword_deleted"],
        },
        (
            "notes.keyword_collection_link",
            organization_link_id(
                "notes.keyword_collection_link",
                (IDS["collection_child"], IDS["keyword_deleted"]),
            ),
        ): {
            "collection_sync_id": IDS["collection_child"],
            "keyword_sync_id": IDS["keyword_deleted"],
        },
        (
            "notes.keyword_collection_link",
            organization_link_id(
                "notes.keyword_collection_link",
                (IDS["collection_remote"], IDS["keyword_deleted"]),
            ),
        ): {
            "collection_sync_id": IDS["collection_remote"],
            "keyword_sync_id": IDS["keyword_deleted"],
        },
        (
            "notes.folder_link",
            organization_link_id(
                "notes.folder_link",
                (IDS["note_enrolled"], IDS["folder_child"]),
            ),
        ): {
            "note_id": IDS["note_enrolled"],
            "folder_sync_id": IDS["folder_child"],
        },
        (
            "notes.folder_link",
            organization_link_id(
                "notes.folder_link",
                (IDS["note_enrolled"], IDS["folder_parent"]),
            ),
        ): {
            "note_id": IDS["note_enrolled"],
            "folder_sync_id": IDS["folder_parent"],
        },
    }
    for (domain, object_id), payload in expected_links.items():
        assert payloads[(domain, "upsert", object_id)] == payload
    kept_local_link_id = organization_link_id(
        "notes.keyword_link",
        ("note", IDS["note_enrolled"], IDS["keyword_keep"]),
    )
    assert not any(
        domain == "notes.keyword_link" and object_id == kept_local_link_id
        for domain, object_id, _operation in logical
    )
    evidenced_deleted_links = {
        (
            "notes.keyword_link",
            organization_link_id(
                "notes.keyword_link",
                ("note", IDS["note_enrolled"], IDS["keyword_deleted"]),
            ),
        ),
        (
            "notes.keyword_collection_link",
            organization_link_id(
                "notes.keyword_collection_link",
                (IDS["collection_remote"], IDS["keyword_deleted"]),
            ),
        ),
        (
            "notes.folder_link",
            organization_link_id(
                "notes.folder_link",
                (IDS["note_enrolled"], IDS["folder_parent"]),
            ),
        ),
        (
            "notes.folder_link",
            organization_link_id(
                "notes.folder_link",
                (IDS["note_enrolled"], IDS["folder_child"]),
            ),
        ),
    }
    for domain, object_id in evidenced_deleted_links:
        assert payloads[(domain, "tombstone", object_id)] == payloads[
            (domain, "upsert", object_id)
        ]
    tombstones = {
        (str(row["domain"]), str(row["object_id"]))
        for row in rows
        if row["operation"] == "tombstone"
    }
    assert tombstones == evidenced_deleted_links | {
        ("notes.keyword", IDS["keyword_deleted"]),
        ("notes.keyword_collection", IDS["collection_remote"]),
        ("notes.folder", IDS["folder_parent"]),
    }

    db = CharactersRAGDB(path, client_id="inventory-tests")
    connection = db.get_connection()
    collection_sync_id = connection.execute(
        "SELECT sync_id FROM keyword_collections WHERE name = 'Deleted collection'"
    ).fetchone()[0]
    keyword_sync_id = connection.execute(
        "SELECT sync_id FROM keywords WHERE keyword = 'Keep local'"
    ).fetchone()[0]
    kept_local_link_count = connection.execute(
        "SELECT COUNT(*) FROM note_keywords AS link JOIN keywords AS keyword "
        "ON keyword.id = link.keyword_id WHERE keyword.sync_id = ?",
        (IDS["keyword_keep"],),
    ).fetchone()[0]
    checkpoint = connection.execute(
        "SELECT inventory_phase, last_inventory_key FROM "
        "notes_organization_sync_checkpoints WHERE server_profile_id = ? AND dataset_id = ?",
        (PROFILE, DATASET),
    ).fetchone()
    db.close_connection()
    assert collection_sync_id == IDS["collection_remote"]
    assert keyword_sync_id == IDS["keyword_keep"]
    assert kept_local_link_count == 1
    assert checkpoint["inventory_phase"] == "complete"
    assert checkpoint["last_inventory_key"] is not None
    assert "private note body" not in json.dumps(rows)


def test_folder_merge_remaps_and_deduplicates_suppressions_before_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "folder-merge.sqlite"
    _seed_inventory_db(path)
    adopted_folder_id = _id(31)
    db = CharactersRAGDB(path, client_id="inventory-tests")
    with db.transaction() as cursor:
        _insert_review(
            cursor,
            review_id="folder-merge-review",
            domain="notes.folder",
            local_object_id="folder-child-local",
            remote_object_id=adopted_folder_id,
            resolution="merge",
        )
        cursor.execute(
            "INSERT INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at) "
            "VALUES (?, ?, ?), (?, ?, ?)",
            (
                IDS["note_missing"],
                IDS["folder_child"],
                NOW,
                IDS["note_missing"],
                adopted_folder_id,
                NOW,
            ),
        )
    db.close_connection()

    result = _run_inventory(
        path,
        enrolled_note_ids={IDS["note_enrolled"], IDS["note_missing"]},
    )

    assert result.status == "complete"
    link_id = organization_link_id(
        "notes.folder_link",
        (IDS["note_enrolled"], adopted_folder_id),
    )
    logical = [
        (str(row["domain"]), str(row["object_id"]), str(row["operation"]))
        for row in _intent_rows(path)
    ]
    upsert = ("notes.folder_link", link_id, "upsert")
    tombstone = ("notes.folder_link", link_id, "tombstone")
    assert logical.index(upsert) < logical.index(tombstone)

    db = CharactersRAGDB(path, client_id="inventory-tests")
    suppressions = [
        tuple(row)
        for row in db.get_connection()
        .execute(
            "SELECT note_id, folder_sync_id FROM note_folder_sync_suppressions "
            "ORDER BY note_id, folder_sync_id"
        )
        .fetchall()
    ]
    db.close_connection()
    assert suppressions == [
        (IDS["note_enrolled"], adopted_folder_id),
        (IDS["note_missing"], adopted_folder_id),
    ]


def test_inventory_resumes_every_commit_without_duplicates_or_skips(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.sqlite"
    _seed_inventory_db(reference_path)
    events: list[tuple[str, str | None]] = []
    reference_result = _run_inventory(
        reference_path,
        after_commit=lambda phase, key: events.append((phase, key)),
    )
    reference = {
        (str(row["domain"]), str(row["object_id"]), str(row["operation"])): (
            str(row["intent_id"]),
            str(row["payload_json"]),
            int(row["source_version"]),
        )
        for row in _intent_rows(reference_path)
    }
    assert events[0] == ("resources", None)
    assert events[-1] == ("complete", None)
    assert len(events) == 22

    class InjectedCrash(RuntimeError):
        pass

    for stop_after in range(len(events)):
        path = tmp_path / f"crash-{stop_after}.sqlite"
        _seed_inventory_db(path)
        seen = 0

        def crash_at_selected_commit(phase: str, key: str | None) -> None:
            nonlocal seen
            if seen == stop_after:
                raise InjectedCrash(f"{phase}:{key}")
            seen += 1

        with pytest.raises(InjectedCrash):
            _run_inventory(path, after_commit=crash_at_selected_commit)
        before_resume = {
            (str(row["domain"]), str(row["object_id"]), str(row["operation"])): str(
                row["intent_id"]
            )
            for row in _intent_rows(path)
        }
        result = _run_inventory(path)
        assert result.status == "complete"
        assert result.skipped_dependencies == reference_result.skipped_dependencies
        resumed_rows = _intent_rows(path)
        resumed = {
            (str(row["domain"]), str(row["object_id"]), str(row["operation"])): (
                str(row["intent_id"]),
                str(row["payload_json"]),
                int(row["source_version"]),
            )
            for row in resumed_rows
        }
        assert resumed == reference
        assert len(resumed_rows) == len(resumed)
        assert all(resumed[key][0] == intent_id for key, intent_id in before_resume.items())


def test_completed_inventory_reopens_with_skips_and_holds_on_later_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "complete-reopen.sqlite"
    _seed_inventory_db(path)

    class InjectedCrash(RuntimeError):
        pass

    with pytest.raises(InjectedCrash):
        _run_inventory(
            path,
            after_commit=lambda phase, key: (_ for _ in ()).throw(InjectedCrash())
            if phase == "complete"
            else None,
        )
    db = CharactersRAGDB(path, client_id="inventory-tests")
    checkpoint = db.get_connection().execute(
        "SELECT inventory_phase, last_inventory_key FROM "
        "notes_organization_sync_checkpoints WHERE server_profile_id = ? AND dataset_id = ?",
        (PROFILE, DATASET),
    ).fetchone()
    db.close_connection()
    assert checkpoint["inventory_phase"] == "complete"
    assert checkpoint["last_inventory_key"] is not None

    result = _run_inventory(path)
    assert result.status == "complete"
    assert result.skipped_dependencies == (
        ("notes.keyword_link", "chat.conversation", "conversation-missing"),
        ("notes.keyword_link", "notes.note", IDS["note_missing"]),
    )
    durable_before = _durable_inventory_bytes(path)

    db = CharactersRAGDB(path, client_id="inventory-tests")
    with db.transaction() as cursor:
        collection_child_id = cursor.execute(
            "SELECT id FROM keyword_collections WHERE name = 'Collection child'"
        ).fetchone()[0]
        keyword_deleted_id = cursor.execute(
            "SELECT id FROM keywords WHERE keyword = 'Deleted keyword'"
        ).fetchone()[0]
        cursor.execute(
            """
            INSERT INTO sync_log(
                entity, entity_id, operation, timestamp, client_id, version, payload
            ) VALUES ('collection_keywords', 'post-complete-evidence', 'delete',
                      ?, 'legacy', 1, ?)
            """,
            (
                NOW,
                json.dumps(
                    {
                        "collection_id": collection_child_id,
                        "keyword_id": keyword_deleted_id,
                    }
                ),
            ),
        )
    db.close_connection()

    drifted = _run_inventory(path)
    assert (drifted.status, drifted.reason_code) == (
        "held",
        "inventory_source_changed",
    )
    assert _durable_inventory_bytes(path) == durable_before


def test_inventory_holds_on_source_drift_instead_of_rebuilding_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drift.sqlite"
    _seed_inventory_db(path)

    class InjectedCrash(RuntimeError):
        pass

    with pytest.raises(InjectedCrash):
        _run_inventory(
            path,
            after_commit=lambda phase, key: (_ for _ in ()).throw(InjectedCrash())
            if phase == "resources" and key is not None
            else None,
        )
    assert len(_intent_rows(path)) == 1
    durable_before = _durable_inventory_bytes(path)
    db = CharactersRAGDB(path, client_id="inventory-tests")
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE keywords SET keyword = 'Newer snapshot value' "
            "WHERE sync_id = ?",
            (IDS["keyword_deleted"],),
        )
    db.close_connection()

    result = _run_inventory(path)

    assert (result.status, result.reason_code) == (
        "held",
        "inventory_source_changed",
    )
    assert _durable_inventory_bytes(path) == durable_before
    assert b"private note body" not in durable_before
    assert b"/Deleted folder" not in durable_before


def test_inventory_holds_on_relationship_evidence_drift_without_durable_writes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence-drift.sqlite"
    _seed_inventory_db(path)

    class InjectedCrash(RuntimeError):
        pass

    with pytest.raises(InjectedCrash):
        _run_inventory(
            path,
            after_commit=lambda phase, key: (_ for _ in ()).throw(InjectedCrash())
            if phase == "resources" and key is not None
            else None,
        )
    durable_before = _durable_inventory_bytes(path)
    db = CharactersRAGDB(path, client_id="inventory-tests")
    with db.transaction() as cursor:
        collection_child_id = cursor.execute(
            "SELECT id FROM keyword_collections WHERE name = 'Collection child'"
        ).fetchone()[0]
        keyword_deleted_id = cursor.execute(
            "SELECT id FROM keywords WHERE keyword = 'Deleted keyword'"
        ).fetchone()[0]
        cursor.execute(
            """
            INSERT INTO sync_log(
                entity, entity_id, operation, timestamp, client_id, version, payload
            ) VALUES ('collection_keywords', 'later-deletion-evidence', 'delete',
                      ?, 'legacy', 1, ?)
            """,
            (
                NOW,
                json.dumps(
                    {
                        "collection_id": collection_child_id,
                        "keyword_id": keyword_deleted_id,
                    }
                ),
            ),
        )
    db.close_connection()

    result = _run_inventory(path)

    assert (result.status, result.reason_code) == (
        "held",
        "inventory_source_changed",
    )
    assert _durable_inventory_bytes(path) == durable_before


def test_inventory_holds_on_dependency_set_drift_without_durable_writes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dependency-drift.sqlite"
    _seed_inventory_db(path)

    class InjectedCrash(RuntimeError):
        pass

    with pytest.raises(InjectedCrash):
        _run_inventory(
            path,
            after_commit=lambda phase, key: (_ for _ in ()).throw(InjectedCrash())
            if phase == "resources" and key is not None
            else None,
        )
    durable_before = _durable_inventory_bytes(path)

    result = _run_inventory(
        path,
        enrolled_note_ids={IDS["note_enrolled"], IDS["note_missing"]},
    )

    assert (result.status, result.reason_code) == (
        "held",
        "inventory_source_changed",
    )
    assert _durable_inventory_bytes(path) == durable_before


def test_parent_first_hierarchy_walk_is_linear_for_deep_trees() -> None:
    from tldw_chatbook.Sync_Interop.notes_organization_inventory import (
        LegacyNotesOrganizationInventory,
        _parent_first,
    )

    accesses = 0

    class CountingRow(dict[str, str | None]):
        def __getitem__(self, key: str) -> str | None:
            nonlocal accesses
            accesses += 1
            return super().__getitem__(key)

    size = 400
    rows = [
        CountingRow(
            id=f"{index:04d}",
            parent_id=None if index == 0 else f"{index - 1:04d}",
        )
        for index in range(size)
    ]

    ordered = _parent_first(rows)  # type: ignore[arg-type]

    assert [row["id"] for row in ordered] == [f"{index:04d}" for index in range(size)]
    assert accesses <= size * 6
    assert "_resource_deleted" not in inspect.getsource(
        LegacyNotesOrganizationInventory
    )


def test_inventory_checkpoint_compare_and_swap_rejects_stale_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stale.sqlite"
    _seed_inventory_db(path)

    class InjectedCrash(RuntimeError):
        pass

    with pytest.raises(InjectedCrash):
        _run_inventory(
            path,
            after_commit=lambda phase, key: (_ for _ in ()).throw(InjectedCrash()),
        )
    db = CharactersRAGDB(path, client_id="inventory-tests")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    with pytest.raises(NotesOrganizationRepositoryError) as exc_info:
        with db.transaction() as cursor:
            repository.advance_inventory_checkpoint(
                cursor,
                dataset_id=DATASET,
                expected_phase="not_started",
                expected_key=None,
                inventory_phase="resources",
                last_inventory_key="stale",
            )
    db.close_connection()
    assert exc_info.value.reason_code == "stale_inventory_checkpoint"


def test_inventory_does_not_republish_applied_remote_state(tmp_path: Path) -> None:
    """A pulled server head is not legacy local state that needs publication."""

    path = tmp_path / "remote-head.sqlite"
    db = CharactersRAGDB(path, client_id="inventory-remote-head")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    object_id = _id(20)
    payload = {"name": "Agent_Lessons", "parent_sync_id": None}
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        with db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO notes_organization_sync_checkpoints(
                    server_profile_id, dataset_id, local_state, server_state,
                    bootstrap_id, captured_count, expected_count, error_code,
                    pull_cursor, inventory_phase, updated_at
                ) VALUES (?, ?, 'adoption_review', 'ready', 'bootstrap-remote',
                          1, 1, NULL, '1', 'not_started', ?)
                """,
                (PROFILE, DATASET, NOW),
            )
            applied = repository.apply_envelope(
                cursor,
                dataset_id=DATASET,
                domain="notes.folder",
                object_id=object_id,
                operation="upsert",
                payload=payload,
                object_revision=1,
                object_hash=payload_hash,
                server_cursor="1",
            )
        assert applied.status == "applied"

        result = _inventory_types()(
            repository,
            dataset_id=DATASET,
            enrolled_note_ids=(),
            enrolled_conversation_ids=(),
        ).run()

        assert result.status == "complete"
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents "
            "WHERE server_profile_id = ? AND dataset_id = ? AND object_id = ?",
            (PROFILE, DATASET, object_id),
        ).fetchone()[0] == 0
    finally:
        db.close_connection()
