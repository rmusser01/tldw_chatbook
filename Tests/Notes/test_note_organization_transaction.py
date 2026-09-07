from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
    NotesOrganizationRepositoryError,
)
from tldw_chatbook.Sync_Interop.notes_organization import organization_link_id


USER_ID = "transaction-user"
PROFILE_ID = "profile-a"
DATASET_ID = "dataset-a"


@pytest.fixture
def loguru_caplog(caplog):
    """Bridge Loguru into pytest's standard log capture for privacy assertions."""

    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    sink_id = loguru_logger.add(PropagateHandler(), format="{message}")
    caplog.set_level(logging.DEBUG)
    try:
        yield caplog
    finally:
        loguru_logger.remove(sink_id)


def _service(tmp_path: Path, *, failure_injector=None):
    db = CharactersRAGDB(db_path=tmp_path / "notes.db", client_id=USER_ID)
    service = NotesInteropService(
        tmp_path,
        "transaction-tests",
        global_db_to_use=db,
        failure_injector=failure_injector,
    )
    service._db_instances[USER_ID] = db
    return service, db


def _checkpoint(db: CharactersRAGDB, *, ready: bool) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES (?, ?, ?, ?, ?, '2026-08-30T00:00:00Z')
            """,
            (
                PROFILE_ID,
                DATASET_ID,
                "ready" if ready else "initializing",
                "ready" if ready else "initializing",
                "complete" if ready else "not_started",
            ),
        )


def _save(service: NotesInteropService, **overrides):
    arguments = {
        "user_id": USER_ID,
        "title": "Atomic lesson",
        "content": "Verified body",
        "folder": "Agent_Lessons",
        "ensure_keywords": ("agent-lesson",),
        "receipt_id": "receipt-atomic-1",
        "server_profile_id": PROFILE_ID,
        "dataset_id": DATASET_ID,
    }
    arguments.update(overrides)
    return service.save_note_with_organization(**arguments)


def _counts(db: CharactersRAGDB) -> dict[str, int]:
    connection = db.get_connection()
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "notes",
            "note_folders",
            "keywords",
            "note_folder_memberships",
            "note_keywords",
            "notes_organization_sync_intents",
            "note_organization_receipts",
        )
    }


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _apply_remote_tombstone(
    db: CharactersRAGDB,
    *,
    domain: str,
    object_id: str,
    payload: dict[str, object],
) -> None:
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE_ID)
    with db.transaction() as cursor:
        result = repository.apply_envelope(
            cursor,
            dataset_id=DATASET_ID,
            domain=domain,
            object_id=object_id,
            operation="tombstone",
            payload=payload,
            object_revision=1,
            object_hash=_payload_hash(payload),
            server_cursor=f"cursor-{domain}",
        )
    assert result.status == "applied"


def _organization_token(db: CharactersRAGDB, note_id: str) -> str:
    with db.transaction() as cursor:
        return str(
            db._library_organization_for_notes(cursor, [note_id])[note_id][
                "organization_version"
            ]
        )


@pytest.mark.parametrize(
    "stage",
    (
        "after_note_write",
        "after_folder_ensure",
        "after_keyword_ensure",
        "after_membership",
        "after_intent",
    ),
)
def test_ready_failure_points_roll_back_the_whole_notes_transaction(tmp_path, stage):
    def fail(point: str) -> None:
        if point == stage:
            raise RuntimeError(stage)

    service, db = _service(tmp_path, failure_injector=fail)
    _checkpoint(db, ready=True)

    with pytest.raises(RuntimeError, match=stage):
        _save(service)

    assert _counts(db) == {
        "notes": 0,
        "note_folders": 0,
        "keywords": 0,
        "note_folder_memberships": 0,
        "note_keywords": 0,
        "notes_organization_sync_intents": 0,
        "note_organization_receipts": 0,
    }
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes'"
    ).fetchone()[0] == 0


def test_pending_receipt_failure_rolls_back_note_receipt_and_note_intent(tmp_path):
    def fail(point: str) -> None:
        if point == "after_receipt":
            raise RuntimeError(point)

    service, db = _service(tmp_path, failure_injector=fail)
    _checkpoint(db, ready=False)

    with pytest.raises(RuntimeError, match="after_receipt"):
        _save(service)

    assert _counts(db)["notes"] == 0
    assert _counts(db)["note_organization_receipts"] == 0
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes'"
    ).fetchone()[0] == 0


def test_failed_atomic_save_rolls_back_without_logging_note_or_keyword_payloads(
    tmp_path, loguru_caplog
):
    secrets = (
        "private-title-marker",
        "private-body-marker",
        "private-keyword-marker",
    )

    def fail(point: str) -> None:
        if point == "after_keyword_ensure":
            raise RuntimeError(point)

    service, db = _service(tmp_path, failure_injector=fail)
    _checkpoint(db, ready=True)

    with pytest.raises(RuntimeError, match="after_keyword_ensure"):
        _save(
            service,
            title=secrets[0],
            content=secrets[1],
            ensure_keywords=(secrets[2],),
            folder=None,
            receipt_id="private-log-rollback",
        )

    assert _counts(db)["notes"] == 0
    assert _counts(db)["keywords"] == 0
    for secret in secrets:
        assert secret not in loguru_caplog.text


def test_ready_save_is_additive_and_update_without_folder_preserves_memberships(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    first = _save(service, ensure_keywords=("existing",), receipt_id="ready-create")
    note_id = first["id"]
    initial_folder_ids = {
        row[0]
        for row in db.get_connection().execute(
            "SELECT folder_id FROM note_folder_memberships WHERE note_id = ? AND deleted = 0",
            (note_id,),
        )
    }
    token = first["organization_version"]

    updated = _save(
        service,
        title="Atomic lesson updated",
        content="Verified body updated",
        note_id=note_id,
        expected_version=1,
        expected_organization_version=token,
        folder=None,
        ensure_keywords=("agent-lesson",),
        receipt_id="ready-update",
    )

    assert updated["version"] == 2
    assert {item["name"] for item in updated["keyword_metadata"]} == {
        "existing",
        "agent-lesson",
    }
    assert {
        row[0]
        for row in db.get_connection().execute(
            "SELECT folder_id FROM note_folder_memberships WHERE note_id = ? AND deleted = 0",
            (note_id,),
        )
    } == initial_folder_ids


def test_ready_save_represents_server_portable_unicode_root_folder(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)

    saved = _save(
        service,
        folder=" Study／Book ",
        ensure_keywords=(),
        receipt_id="portable-unicode-folder",
    )

    assert saved["receipt_state"] is None
    assert saved["organization_state"] == "ready"
    assert saved["folders"][0]["name"] == "Study／Book"
    row = db.get_connection().execute(
        "SELECT parent_id, name, normalized_name, path, normalized_path "
        "FROM note_folders WHERE sync_id = ?",
        (saved["folders"][0]["id"],),
    ).fetchone()
    assert tuple(row) == (
        None,
        "Study／Book",
        "study/book",
        "/Study／Book",
        "/study/book",
    )
    found = service.search_library_notes(
        USER_ID, folder="Study／Book", limit=20, offset=0
    )
    assert [item["id"] for item in found["items"]] == [saved["id"]]


@pytest.mark.parametrize(
    "folder",
    ("", ".", "..", "Study/Book", "Study\\Book", "Study\x00Book"),
)
def test_save_rejects_invalid_portable_root_folder_without_writes(tmp_path, folder):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    before = _counts(db)

    with pytest.raises(NotesOrganizationRepositoryError) as rejected:
        _save(service, folder=folder, receipt_id="invalid-portable-folder")

    assert rejected.value.reason_code == "invalid_name"
    assert _counts(db) == before


def test_pending_save_preserves_trimmed_portable_unicode_folder_request(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=False)

    saved = _save(
        service,
        folder=" Study／Book ",
        receipt_id="pending-portable-unicode",
    )

    assert saved["receipt_state"] == "pending_organization"
    assert saved["folders"] == []
    receipt = db.get_connection().execute(
        "SELECT requested_folder_name FROM note_organization_receipts "
        "WHERE receipt_id = 'pending-portable-unicode'"
    ).fetchone()
    assert receipt["requested_folder_name"] == "Study／Book"


def test_portable_unicode_root_folder_local_path_collision_becomes_review(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    folders = LocalNoteFolderRepository(db)
    parent = folders.create_folder(name="Study", parent_id=None)
    collision = folders.create_folder(name="Book", parent_id=parent.folder_id)

    saved = _save(
        service,
        folder="Study／Book",
        ensure_keywords=("agent-lesson",),
        receipt_id="portable-unicode-review",
    )

    assert saved["receipt_state"] == "placement_review"
    assert saved["organization_state"] == "placement_review"
    assert saved["folders"] == []
    receipt = db.get_connection().execute(
        "SELECT collision_ids_json FROM note_organization_receipts "
        "WHERE receipt_id = 'portable-unicode-review'"
    ).fetchone()
    assert json.loads(receipt["collision_ids_json"]) == [collision.folder_id]


def test_stale_content_and_organization_versions_refuse_without_writes(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    first = _save(service, receipt_id="stale-create")
    note_id = first["id"]
    before = _counts(db)

    with pytest.raises(ConflictError):
        _save(
            service,
            note_id=note_id,
            expected_version=99,
            expected_organization_version=first["organization_version"],
            folder=None,
            receipt_id="stale-note",
        )
    assert _counts(db) == before

    with db.transaction() as cursor:
        keyword_id = db.add_keyword("concurrent", cursor=cursor)
        db.link_note_to_keyword(note_id, int(keyword_id), cursor=cursor)
    after_concurrent_change = _counts(db)
    with pytest.raises(NotesOrganizationRepositoryError) as error:
        _save(
            service,
            note_id=note_id,
            expected_version=1,
            expected_organization_version=first["organization_version"],
            folder=None,
            receipt_id="stale-organization",
        )
    assert error.value.reason_code == "organization_changed"
    assert _counts(db) == after_concurrent_change


def test_concurrent_folder_and_keyword_ensure_converges_without_duplicates(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[BaseException] = []

    def save(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                _save(
                    service,
                    title=f"Concurrent {index}",
                    receipt_id=f"concurrent-{index}",
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(results) == 2
    connection = db.get_connection()
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folders WHERE deleted = 0 AND name = 'Agent_Lessons'"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM keywords WHERE deleted = 0 AND keyword = 'agent-lesson'"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folder_memberships WHERE deleted = 0"
    ).fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM note_keywords").fetchone()[0] == 2


def test_new_resources_and_links_emit_ordinary_upserts_without_restore_metadata(
    tmp_path,
):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)

    _save(service, receipt_id="ordinary-upserts")

    rows = db.get_connection().execute(
        "SELECT domain, routing_metadata_json "
        "FROM notes_organization_sync_intents ORDER BY intent_sequence"
    ).fetchall()
    assert {str(row["domain"]) for row in rows} == {
        "notes.folder",
        "notes.keyword",
        "notes.folder_link",
        "notes.keyword_link",
    }
    assert {str(row["routing_metadata_json"]) for row in rows} == {"{}"}


def test_restored_keyword_resource_marks_only_its_exact_tombstone_successor(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    keyword_id = db.add_keyword("agent-lesson")
    keyword = db.get_connection().execute(
        "SELECT sync_id FROM keywords WHERE id = ?", (keyword_id,)
    ).fetchone()
    _apply_remote_tombstone(
        db,
        domain="notes.keyword",
        object_id=str(keyword["sync_id"]),
        payload={},
    )

    _save(service, folder=None, receipt_id="resource-restore")

    rows = db.get_connection().execute(
        "SELECT domain, routing_metadata_json "
        "FROM notes_organization_sync_intents ORDER BY intent_sequence"
    ).fetchall()
    assert [(row["domain"], row["routing_metadata_json"]) for row in rows] == [
        ("notes.keyword", '{"restore_intent":true}'),
        ("notes.keyword_link", "{}"),
    ]


def test_restored_keyword_link_marks_only_its_exact_tombstone_successor(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    note_id = db.add_note("Existing note", "Existing body")
    keyword_id = db.add_keyword("agent-lesson")
    keyword = db.get_connection().execute(
        "SELECT sync_id FROM keywords WHERE id = ?", (keyword_id,)
    ).fetchone()
    db.link_note_to_keyword(str(note_id), int(keyword_id))
    payload = {
        "subject_type": "note",
        "subject_id": str(note_id),
        "keyword_sync_id": str(keyword["sync_id"]),
    }
    object_id = organization_link_id(
        "notes.keyword_link",
        ("note", str(note_id), str(keyword["sync_id"])),
    )
    _apply_remote_tombstone(
        db,
        domain="notes.keyword_link",
        object_id=object_id,
        payload=payload,
    )

    _save(
        service,
        note_id=str(note_id),
        expected_version=1,
        expected_organization_version=_organization_token(db, str(note_id)),
        folder=None,
        receipt_id="keyword-link-restore",
    )

    row = db.get_connection().execute(
        "SELECT routing_metadata_json FROM notes_organization_sync_intents "
        "WHERE domain = 'notes.keyword_link' AND object_id = ?",
        (object_id,),
    ).fetchone()
    assert row["routing_metadata_json"] == '{"restore_intent":true}'


def test_restored_folder_link_marks_only_its_exact_tombstone_successor(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    note_id = db.add_note("Existing note", "Existing body")
    folders = LocalNoteFolderRepository(db)
    folder = folders.create_folder(name="Agent_Lessons", parent_id=None)
    folder_row = db.get_connection().execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (folder.folder_id,)
    ).fetchone()
    folders.attach_manual(folder_id=folder.folder_id, note_id=str(note_id))
    payload = {
        "note_id": str(note_id),
        "folder_sync_id": str(folder_row["sync_id"]),
    }
    object_id = organization_link_id(
        "notes.folder_link", (str(note_id), str(folder_row["sync_id"]))
    )
    _apply_remote_tombstone(
        db,
        domain="notes.folder_link",
        object_id=object_id,
        payload=payload,
    )

    _save(
        service,
        note_id=str(note_id),
        expected_version=1,
        expected_organization_version=_organization_token(db, str(note_id)),
        ensure_keywords=(),
        receipt_id="folder-link-restore",
    )

    row = db.get_connection().execute(
        "SELECT routing_metadata_json FROM notes_organization_sync_intents "
        "WHERE domain = 'notes.folder_link' AND object_id = ?",
        (object_id,),
    ).fetchone()
    assert row["routing_metadata_json"] == '{"restore_intent":true}'


def test_pending_save_is_discoverable_without_links_or_publishable_intents(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=False)

    saved = _save(service, receipt_id="pending-create")

    assert saved["receipt_state"] == "pending_organization"
    assert saved["organization_state"] == "pending"
    assert saved["keyword_metadata"] == []
    assert saved["folders"] == []
    connection = db.get_connection()
    receipt = connection.execute(
        "SELECT * FROM note_organization_receipts WHERE receipt_id = 'pending-create'"
    ).fetchone()
    assert receipt is not None
    requested_receipt_data = json.loads(str(receipt["requested_keywords_json"]))
    assert requested_receipt_data[:-1] == ["agent-lesson"]
    assert set(requested_receipt_data[-1]) == {"_request"}
    assert "Atomic lesson" not in str(dict(receipt))
    assert "Verified body" not in str(dict(receipt))
    assert connection.execute("SELECT COUNT(*) FROM note_keywords").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folder_memberships"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
        (saved["id"],),
    ).fetchone()[0] == 0
    search = service.search_library_notes(
        USER_ID, keyword="agent-lesson", limit=10, offset=0
    )
    assert [item["id"] for item in search["items"]] == [saved["id"]]
    assert search["items"][0]["organization_state"] == "pending"


def test_reusing_pending_receipt_returns_the_same_note_without_duplicate_create(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=False)
    first = _save(service, receipt_id="retry-receipt")

    retried = _save(service, receipt_id="retry-receipt")

    assert retried["id"] == first["id"]
    assert retried["version"] == first["version"] == 1
    assert db.get_connection().execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM note_organization_receipts"
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "changed_field",
    (
        "title",
        "content",
        "note_id",
        "expected_version",
        "expected_organization_version",
        "profile",
        "dataset",
        "folder",
        "keywords",
    ),
)
def test_receipt_replay_rejects_any_changed_request_identity(
    tmp_path, changed_field: str
) -> None:
    service, db = _service(tmp_path)
    _checkpoint(db, ready=False)
    note_id = str(db.add_note("Before", "Before"))
    initial_organization_version = _organization_token(db, note_id)
    arguments = {
        "note_id": note_id,
        "expected_version": 1,
        "expected_organization_version": initial_organization_version,
        "receipt_id": "request-identity",
    }
    first = _save(service, **arguments)

    retry_arguments = dict(arguments)
    if changed_field == "title":
        retry_arguments["title"] = "Changed title"
    elif changed_field == "content":
        retry_arguments["content"] = "Changed body"
    elif changed_field == "note_id":
        other_note_id = str(db.add_note("Other", "Other"))
        retry_arguments["note_id"] = other_note_id
        retry_arguments["expected_organization_version"] = _organization_token(
            db, other_note_id
        )
    elif changed_field == "expected_version":
        retry_arguments["expected_version"] = 2
    elif changed_field == "expected_organization_version":
        retry_arguments["expected_organization_version"] = first[
            "organization_version"
        ]
    elif changed_field == "profile":
        retry_arguments["server_profile_id"] = "profile-b"
    elif changed_field == "dataset":
        retry_arguments["dataset_id"] = "dataset-b"
    elif changed_field == "folder":
        retry_arguments["folder"] = "Other_Folder"
    else:
        retry_arguments["ensure_keywords"] = ("agent-lesson", "changed")

    with pytest.raises(NotesOrganizationRepositoryError) as error:
        _save(service, **retry_arguments)

    assert error.value.reason_code == "receipt_conflict"
    persisted = db.get_connection().execute(
        "SELECT title, content, version FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    assert tuple(persisted) == ("Atomic lesson", "Verified body", 2)


def test_pending_receipt_request_binding_is_content_free_and_exact_retry_is_stable(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    _checkpoint(db, ready=False)

    first = _save(service, receipt_id="content-free-binding")
    retried = _save(service, receipt_id="content-free-binding")

    assert retried == first
    receipt = db.get_connection().execute(
        "SELECT * FROM note_organization_receipts "
        "WHERE receipt_id = 'content-free-binding'"
    ).fetchone()
    serialized = str(dict(receipt))
    assert "Atomic lesson" not in serialized
    assert "Verified body" not in serialized
    request_data = json.loads(str(receipt["requested_keywords_json"]))[-1][
        "_request"
    ]
    assert request_data["server_profile_id"] == PROFILE_ID
    assert request_data["dataset_id"] == DATASET_ID
    assert request_data["fingerprint"] == request_data["fingerprint"].lower()
    assert len(request_data["fingerprint"]) == 64


def _create_unresolved_note(
    service: NotesInteropService,
    db: CharactersRAGDB,
    *,
    receipt_state: str,
) -> dict:
    if receipt_state == "pending_organization":
        _checkpoint(db, ready=False)
    else:
        _checkpoint(db, ready=True)
        LocalNoteFolderRepository(db).create_folder(
            name="agent_lessons", parent_id=None
        )
    saved = _save(service, receipt_id=f"original-{receipt_state}")
    assert saved["receipt_state"] == receipt_state
    return saved


def _folder_sync_id(db: CharactersRAGDB, folder_id: str) -> str:
    row = db.get_connection().execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (folder_id,)
    ).fetchone()
    assert row is not None and row["sync_id"]
    return str(row["sync_id"])


def _review_rows(db: CharactersRAGDB) -> list[dict]:
    rows = db.get_connection().execute(
        "SELECT * FROM notes_organization_adoption_reviews "
        "WHERE server_profile_id = ? AND dataset_id = ? "
        "ORDER BY created_at, review_id",
        (PROFILE_ID, DATASET_ID),
    ).fetchall()
    return [dict(row) for row in rows]


def _receipt_keywords(receipt: dict | sqlite3.Row) -> list[str]:
    return [
        item
        for item in json.loads(str(receipt["requested_keywords_json"]))
        if isinstance(item, str)
    ]


def test_pending_content_only_edit_inherits_desired_organization_and_replays(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(
        service, db, receipt_state="pending_organization"
    )
    arguments = {
        "user_id": USER_ID,
        "title": "Pending content-only title",
        "content": "Pending content-only body",
        "note_id": first["id"],
        "expected_version": 1,
        "expected_organization_version": first["organization_version"],
        "server_profile_id": PROFILE_ID,
        "dataset_id": DATASET_ID,
    }

    updated = service.save_note_with_organization(**arguments)
    retried = service.save_note_with_organization(**arguments)

    assert retried == updated
    assert updated["version"] == 2
    assert updated["receipt_state"] == "pending_organization"
    assert updated["organization_state"] == "pending"
    receipt = db.get_connection().execute(
        "SELECT * FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchone()
    assert receipt["receipt_id"] == "original-pending_organization"
    assert receipt["note_version"] == 2
    assert receipt["requested_folder_name"] == "Agent_Lessons"
    assert _receipt_keywords(receipt) == ["agent-lesson"]
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
        (first["id"],),
    ).fetchone()[0] == 0
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0


def test_pending_keyword_only_edit_preserves_folder_and_adds_desired_keyword(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(
        service, db, receipt_state="pending_organization"
    )

    updated = service.save_note_with_organization(
        USER_ID,
        title="Pending keyword title",
        content="Pending keyword body",
        note_id=first["id"],
        expected_version=1,
        ensure_keywords=("second-keyword",),
        expected_organization_version=first["organization_version"],
        server_profile_id=PROFILE_ID,
        dataset_id=DATASET_ID,
    )

    assert updated["receipt_state"] == "pending_organization"
    receipt = db.get_connection().execute(
        "SELECT * FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchone()
    assert receipt["requested_folder_name"] == "Agent_Lessons"
    assert _receipt_keywords(receipt) == ["agent-lesson", "second-keyword"]
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM note_keywords WHERE note_id = ?", (first["id"],)
    ).fetchone()[0] == 0
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
        (first["id"],),
    ).fetchone()[0] == 0


def test_placement_content_only_edit_preserves_receipt_review_and_replays(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state="placement_review")
    before = dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    )
    arguments = {
        "user_id": USER_ID,
        "title": "Placement content-only title",
        "content": "Placement content-only body",
        "note_id": first["id"],
        "expected_version": 1,
        "expected_organization_version": first["organization_version"],
        "server_profile_id": PROFILE_ID,
        "dataset_id": DATASET_ID,
    }

    updated = service.save_note_with_organization(**arguments)
    retried = service.save_note_with_organization(**arguments)

    assert retried == updated
    assert updated["receipt_state"] == "placement_review"
    assert updated["organization_state"] == "placement_review"
    receipt = db.get_connection().execute(
        "SELECT * FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchone()
    assert receipt["receipt_id"] == before["receipt_id"]
    assert receipt["review_id"] == before["review_id"]
    assert receipt["note_version"] == 2
    assert receipt["requested_folder_name"] == "Agent_Lessons"
    assert _receipt_keywords(receipt) == ["agent-lesson"]
    review = db.get_connection().execute(
        "SELECT state, resolution FROM notes_organization_adoption_reviews "
        "WHERE review_id = ?",
        (before["review_id"],),
    ).fetchone()
    assert tuple(review) == ("open", None)


@pytest.mark.parametrize(
    "receipt_state", ("pending_organization", "placement_review")
)
def test_omitted_organization_edit_failure_rolls_back_receipt_and_note(
    tmp_path, receipt_state: str
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state=receipt_state)
    connection = db.get_connection()
    before_note = tuple(
        connection.execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        ).fetchone()
    )
    before_receipt = dict(
        connection.execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        ).fetchone()
    )

    def fail(point: str) -> None:
        if point == "after_receipt":
            raise RuntimeError(point)

    service._organization_failure_injector = fail
    with pytest.raises(RuntimeError, match="after_receipt"):
        service.save_note_with_organization(
            USER_ID,
            title="Rollback omitted title",
            content="Rollback omitted body",
            note_id=first["id"],
            expected_version=1,
            expected_organization_version=first["organization_version"],
            server_profile_id=PROFILE_ID,
            dataset_id=DATASET_ID,
        )

    assert tuple(
        connection.execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        ).fetchone()
    ) == before_note
    assert dict(
        connection.execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        ).fetchone()
    ) == before_receipt
    if receipt_state == "pending_organization":
        assert connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
            (first["id"],),
        ).fetchone()[0] == 0


def test_placement_review_update_to_ready_retires_receipt_and_rejects_stale_replay(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state="placement_review")
    original_receipt = dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    )
    valid_folder = LocalNoteFolderRepository(db).create_folder(
        name="Resolved_Lessons", parent_id=None
    )
    valid_folder_sync_id = _folder_sync_id(db, valid_folder.folder_id)
    update = {
        "note_id": first["id"],
        "expected_version": 1,
        "expected_organization_version": first["organization_version"],
        "title": "Resolved placement title",
        "content": "Resolved placement body",
        "folder": None,
        "folder_sync_id": valid_folder_sync_id,
        "receipt_id": None,
    }

    updated = _save(service, **update)
    with pytest.raises(ConflictError):
        _save(service, **update)
    invalid_organization = dict(update)
    invalid_organization["expected_organization_version"] = "0" * 64
    with pytest.raises(ConflictError):
        _save(service, **invalid_organization)
    changed_request = dict(update)
    changed_request["content"] = "Different stale body"
    with pytest.raises(ConflictError):
        _save(service, **changed_request)
    changed_scope = dict(update)
    changed_scope.update(
        server_profile_id="profile-b",
        dataset_id="dataset-b",
    )
    with pytest.raises(ConflictError):
        _save(service, **changed_scope)

    assert updated["version"] == 2
    assert updated["receipt_state"] is None
    assert updated["organization_state"] == "ready"
    assert [folder["id"] for folder in updated["folders"]] == [valid_folder_sync_id]
    connection = db.get_connection()
    assert connection.execute(
        "SELECT COUNT(*) FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchone()[0] == 0
    review = connection.execute(
        "SELECT * FROM notes_organization_adoption_reviews WHERE review_id = ?",
        (original_receipt["review_id"],),
    ).fetchone()
    assert review["state"] == "resolved"
    assert review["resolution"] == "keep_local"
    assert review["resolved_at"] is not None
    serialized_review = json.dumps(dict(review), sort_keys=True)
    assert "Resolved placement title" not in serialized_review
    assert "Resolved placement body" not in serialized_review
    assert connection.execute(
        "SELECT COUNT(*) FROM notes_organization_adoption_reviews "
        "WHERE server_profile_id = ? AND dataset_id = ? AND state = 'open'",
        (PROFILE_ID, DATASET_ID),
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folder_memberships "
        "WHERE note_id = ? AND folder_id = ? AND deleted = 0",
        (first["id"], valid_folder.folder_id),
    ).fetchone()[0] == 1


def test_unrelated_historical_review_cannot_authorize_stale_matching_note_state(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    saved = _save(service, receipt_id="ordinary-ready-save")
    connection = db.get_connection()
    folder = connection.execute(
        "SELECT id FROM note_folders WHERE name = 'Agent_Lessons'"
    ).fetchone()
    with db.transaction() as cursor:
        db._update_note_with_cursor(
            cursor,
            note_id=saved["id"],
            update_data={"title": "Atomic lesson", "content": "Verified body"},
            expected_version=1,
        )
        cursor.execute(
            "INSERT INTO notes_organization_adoption_reviews("
            "review_id, server_profile_id, dataset_id, domain, local_object_id, "
            "remote_object_id, collision_key, display_name, portable_path, state, "
            "resolution, created_at, updated_at, resolved_at) VALUES ("
            "'unrelated-history', ?, ?, 'notes.folder', ?, NULL, "
            "'unrelated', 'Unrelated', 'Unrelated', 'resolved', 'keep_local', "
            "'2026-08-30T00:00:00Z', '2026-08-30T00:00:00Z', "
            "'2026-08-30T00:00:00Z')",
            (PROFILE_ID, DATASET_ID, folder["id"]),
        )

    stale = {
        "note_id": saved["id"],
        "expected_version": 1,
        "expected_organization_version": saved["organization_version"],
        "receipt_id": None,
    }
    with pytest.raises(ConflictError):
        _save(service, **stale)
    stale["expected_organization_version"] = "0" * 64
    with pytest.raises(ConflictError):
        _save(service, **stale)

    review = connection.execute(
        "SELECT * FROM notes_organization_adoption_reviews "
        "WHERE review_id = 'unrelated-history'"
    ).fetchone()
    serialized_review = json.dumps(dict(review), sort_keys=True)
    assert "Atomic lesson" not in serialized_review
    assert "Verified body" not in serialized_review


def test_placement_review_update_keeps_review_open_when_another_receipt_uses_it(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state="placement_review")
    second = _save(service, receipt_id="second-placement-receipt")
    first_receipt = db.get_connection().execute(
        "SELECT review_id FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchone()
    second_receipt = db.get_connection().execute(
        "SELECT review_id FROM note_organization_receipts WHERE note_id = ?",
        (second["id"],),
    ).fetchone()
    assert first_receipt["review_id"] == second_receipt["review_id"]
    valid_folder = LocalNoteFolderRepository(db).create_folder(
        name="Resolved_Lessons", parent_id=None
    )

    _save(
        service,
        note_id=first["id"],
        expected_version=1,
        expected_organization_version=first["organization_version"],
        title="First placement resolved",
        content="First placement resolved body",
        folder=None,
        folder_sync_id=_folder_sync_id(db, valid_folder.folder_id),
        receipt_id=None,
    )

    review = db.get_connection().execute(
        "SELECT state, resolution FROM notes_organization_adoption_reviews "
        "WHERE review_id = ?",
        (second_receipt["review_id"],),
    ).fetchone()
    assert tuple(review) == ("open", None)
    with db.transaction() as cursor:
        remaining = db._library_organization_for_notes(cursor, [second["id"]])[
            second["id"]
        ]
    assert remaining["organization_state"] == "placement_review"


def test_placement_review_update_to_new_collision_moves_single_open_review_and_replays(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state="placement_review")
    original_receipt = dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    )
    collision_b = LocalNoteFolderRepository(db).create_folder(
        name="other_lessons", parent_id=None
    )
    update = {
        "note_id": first["id"],
        "expected_version": 1,
        "expected_organization_version": first["organization_version"],
        "title": "Second collision title",
        "content": "Second collision body",
        "folder": "Other_Lessons",
        "receipt_id": None,
    }

    updated = _save(service, **update)
    retried = _save(service, **update)

    assert retried == updated
    assert updated["version"] == 2
    assert updated["receipt_state"] == "placement_review"
    assert updated["organization_state"] == "placement_review"
    receipt = db.get_connection().execute(
        "SELECT * FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchone()
    assert receipt["receipt_id"] == original_receipt["receipt_id"]
    assert receipt["review_id"] != original_receipt["review_id"]
    assert json.loads(receipt["collision_ids_json"]) == [collision_b.folder_id]
    reviews = _review_rows(db)
    old_review = next(
        row for row in reviews if row["review_id"] == original_receipt["review_id"]
    )
    assert (old_review["state"], old_review["resolution"]) == (
        "resolved",
        "keep_local",
    )
    assert sum(row["state"] == "open" for row in reviews) == 1
    assert [row["review_id"] for row in reviews if row["state"] == "open"] == [
        receipt["review_id"]
    ]


@pytest.mark.parametrize("transition", ("ready", "new_collision"))
def test_placement_review_transition_failure_restores_receipt_note_and_reviews(
    tmp_path, transition: str
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state="placement_review")
    folder_arguments: dict[str, object]
    if transition == "ready":
        valid_folder = LocalNoteFolderRepository(db).create_folder(
            name="Resolved_Lessons", parent_id=None
        )
        folder_arguments = {
            "folder": None,
            "folder_sync_id": _folder_sync_id(db, valid_folder.folder_id),
        }
    else:
        LocalNoteFolderRepository(db).create_folder(
            name="other_lessons", parent_id=None
        )
        folder_arguments = {"folder": "Other_Lessons"}
    connection = db.get_connection()
    before_note = tuple(
        connection.execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        ).fetchone()
    )
    before_receipt = dict(
        connection.execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        ).fetchone()
    )
    before_reviews = _review_rows(db)

    def fail(point: str) -> None:
        if point == "after_receipt":
            raise RuntimeError(point)

    service._organization_failure_injector = fail
    with pytest.raises(RuntimeError, match="after_receipt"):
        _save(
            service,
            note_id=first["id"],
            expected_version=1,
            expected_organization_version=first["organization_version"],
            title="Rolled back transition title",
            content="Rolled back transition body",
            receipt_id=None,
            **folder_arguments,
        )

    assert tuple(
        connection.execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        ).fetchone()
    ) == before_note
    assert dict(
        connection.execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        ).fetchone()
    ) == before_receipt
    assert _review_rows(db) == before_reviews


@pytest.mark.parametrize(
    "receipt_state", ("pending_organization", "placement_review")
)
def test_fresh_update_reuses_unresolved_receipt_and_exact_retry_without_receipt_id(
    tmp_path, receipt_state: str
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state=receipt_state)
    update = {
        "note_id": first["id"],
        "expected_version": 1,
        "expected_organization_version": first["organization_version"],
        "title": "Edited unresolved title",
        "content": "Edited unresolved body",
        "receipt_id": None,
    }

    updated = _save(service, **update)
    retried = _save(service, **update)

    assert retried == updated
    assert updated["id"] == first["id"]
    assert updated["version"] == 2
    assert updated["receipt_state"] == receipt_state
    rows = db.get_connection().execute(
        "SELECT receipt_id, note_version, state "
        "FROM note_organization_receipts WHERE note_id = ?",
        (first["id"],),
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (f"original-{receipt_state}", 2, receipt_state)
    ]
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM notes WHERE id = ?", (first["id"],)
    ).fetchone()[0] == 1


def test_pending_update_stays_dispatcher_excluded_until_explicit_finalization(
    tmp_path,
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(
        service, db, receipt_state="pending_organization"
    )
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET "
            "local_state = 'ready', server_state = 'ready', "
            "inventory_phase = 'complete'"
        )

    updated = _save(
        service,
        note_id=first["id"],
        expected_version=1,
        expected_organization_version=first["organization_version"],
        title="Edited while awaiting finalization",
        content="Still local pending content",
        receipt_id=None,
    )

    assert updated["receipt_state"] == "pending_organization"
    assert updated["organization_state"] == "pending"
    connection = db.get_connection()
    assert connection.execute(
        "SELECT COUNT(*) FROM note_keywords WHERE note_id = ?", (first["id"],)
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ?",
        (first["id"],),
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
        (first["id"],),
    ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("receipt_state", "stale_precondition"),
    (
        ("pending_organization", "note"),
        ("pending_organization", "organization"),
        ("placement_review", "note"),
        ("placement_review", "organization"),
    ),
)
def test_unresolved_receipt_update_rejects_stale_preconditions_without_writes(
    tmp_path, receipt_state: str, stale_precondition: str
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state=receipt_state)
    expected_version = 1
    expected_organization_version = first["organization_version"]
    if stale_precondition == "note":
        expected_version = 99
        expected_error = ConflictError
    else:
        with db.transaction() as cursor:
            keyword_id = db.add_keyword("concurrent-change", cursor=cursor)
            db.link_note_to_keyword(first["id"], int(keyword_id), cursor=cursor)
        expected_error = NotesOrganizationRepositoryError
    before_note = tuple(
        db.get_connection()
        .execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        )
        .fetchone()
    )
    before_receipt = dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    )

    with pytest.raises(expected_error) as error:
        service.save_note_with_organization(
            USER_ID,
            note_id=first["id"],
            expected_version=expected_version,
            expected_organization_version=expected_organization_version,
            title="Rejected stale title",
            content="Rejected stale body",
            server_profile_id=PROFILE_ID,
            dataset_id=DATASET_ID,
        )

    if stale_precondition == "organization":
        assert error.value.reason_code == "organization_changed"
    assert tuple(
        db.get_connection()
        .execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        )
        .fetchone()
    ) == before_note
    assert dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    ) == before_receipt


@pytest.mark.parametrize(
    "receipt_state", ("pending_organization", "placement_review")
)
def test_unresolved_receipt_update_failure_rolls_back_note_and_receipt(
    tmp_path, receipt_state: str
) -> None:
    service, db = _service(tmp_path)
    first = _create_unresolved_note(service, db, receipt_state=receipt_state)
    before_note = tuple(
        db.get_connection()
        .execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        )
        .fetchone()
    )
    before_receipt = dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    )

    def fail(point: str) -> None:
        if point == "after_receipt":
            raise RuntimeError(point)

    service._organization_failure_injector = fail
    with pytest.raises(RuntimeError, match="after_receipt"):
        _save(
            service,
            note_id=first["id"],
            expected_version=1,
            expected_organization_version=first["organization_version"],
            title="Rolled back title",
            content="Rolled back body",
            receipt_id=None,
        )

    assert tuple(
        db.get_connection()
        .execute(
            "SELECT title, content, version FROM notes WHERE id = ?", (first["id"],)
        )
        .fetchone()
    ) == before_note
    assert dict(
        db.get_connection()
        .execute(
            "SELECT * FROM note_organization_receipts WHERE note_id = ?",
            (first["id"],),
        )
        .fetchone()
    ) == before_receipt


def test_exact_keyword_identity_conflict_stays_pending_even_when_group_is_ready(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    db.add_keyword("Agent-Lesson")

    saved = _save(service, folder=None, receipt_id="keyword-review")

    assert saved["receipt_state"] == "pending_organization"
    assert db.get_connection().execute("SELECT COUNT(*) FROM note_keywords").fetchone()[0] == 0
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0


def test_keyword_identity_review_from_another_profile_does_not_block_save(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_adoption_reviews(
                review_id, server_profile_id, dataset_id, domain,
                local_object_id, remote_object_id, collision_key, display_name,
                portable_path, state, created_at, updated_at
            ) VALUES (
                'review-other-profile', 'profile-b', 'dataset-b',
                'notes.keyword', 'keyword-other-profile', NULL,
                'agent-lesson', 'agent-lesson', NULL, 'open', 'now', 'now'
            )
            """
        )

    saved = _save(service, folder=None, receipt_id="profile-scoped-review")

    assert saved["receipt_state"] is None
    assert saved["organization_state"] == "ready"
    assert {item["name"] for item in saved["keyword_metadata"]} == {
        "agent-lesson"
    }


def test_folder_only_collision_records_nonblocking_placement_review(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    folders = LocalNoteFolderRepository(db)
    existing = folders.create_folder(name="Agent_Lessons", parent_id=None)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE note_folders SET sync_id = NULL WHERE id = ?", (existing.folder_id,)
        )

    saved = _save(service, receipt_id="folder-review")

    assert saved["receipt_state"] == "placement_review"
    assert saved["organization_state"] == "placement_review"
    assert {item["name"] for item in saved["keyword_metadata"]} == {"agent-lesson"}
    receipt = db.get_connection().execute(
        "SELECT review_id, collision_ids_json FROM note_organization_receipts "
        "WHERE receipt_id = 'folder-review'"
    ).fetchone()
    assert receipt["review_id"]
    assert existing.folder_id in receipt["collision_ids_json"]
    assert db.get_connection().execute("SELECT COUNT(*) FROM note_keywords").fetchone()[0] == 1


def test_casefold_equivalent_folder_spelling_requires_placement_review(tmp_path):
    service, db = _service(tmp_path)
    _checkpoint(db, ready=True)
    folders = LocalNoteFolderRepository(db)
    existing = folders.create_folder(name="agent_lessons", parent_id=None)

    saved = _save(service, receipt_id="folder-spelling-review")

    assert saved["receipt_state"] == "placement_review"
    assert saved["folders"] == []
    assert {item["name"] for item in saved["keyword_metadata"]} == {
        "agent-lesson"
    }
    connection = db.get_connection()
    assert connection.execute(
        "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ? AND deleted = 0",
        (saved["id"],),
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT collision_ids_json FROM note_organization_receipts "
        "WHERE receipt_id = 'folder-spelling-review'"
    ).fetchone()[0] == json.dumps([existing.folder_id], separators=(",", ":"))
    assert {
        str(row["domain"])
        for row in connection.execute(
            "SELECT domain FROM notes_organization_sync_intents"
        ).fetchall()
    } == {"notes.keyword", "notes.keyword_link"}
