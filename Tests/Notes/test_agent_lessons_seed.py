from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.agent_lessons import (
    AGENT_LESSONS_FOLDER,
    agent_lessons_seed_fingerprint,
    initialize_agent_lessons_folder,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.tldw_api import SyncV2Envelope


PROFILE = "server-a"
DATASET = "dataset-a"


def _db(path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(path, client_id="agent-lessons-seed")


def _seed_row(db: CharactersRAGDB, profile: str, dataset: str):
    return db.get_connection().execute(
        "SELECT * FROM agent_lessons_seed_state WHERE profile_id = ? AND dataset_id = ?",
        (profile, dataset),
    ).fetchone()


def _folder_envelope(object_id: str, *, name: str, revision: int, cursor: int):
    payload = {"name": name, "parent_sync_id": None}
    content = json.dumps(
        {"operation": "upsert", "payload": payload, "revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SyncV2Envelope(
        client_envelope_id=f"remote:{object_id}:{revision}",
        dataset_id=DATASET,
        device_id="remote-device",
        domain="notes.folder",
        object_id=object_id,
        operation="upsert",
        schema_version=1,
        object_revision=revision,
        server_cursor=cursor,
        payload=payload,
        payload_hash=hashlib.sha256(content).hexdigest(),
        encryption_policy="server_trusted_v1",
        routing_metadata={},
    )


def test_local_only_seed_is_atomic_idempotent_and_creates_no_marker(tmp_path: Path):
    db = _db(tmp_path / "local.sqlite")
    try:
        first = initialize_agent_lessons_folder(
            db,
            scope_mode="local_only",
            profile_id="local",
            dataset_id="local",
        )
        second = initialize_agent_lessons_folder(
            db,
            scope_mode="local_only",
            profile_id="local",
            dataset_id="local",
        )

        folders = db.get_connection().execute(
            "SELECT * FROM note_folders WHERE name = ? COLLATE BINARY AND deleted = 0",
            (AGENT_LESSONS_FOLDER,),
        ).fetchall()
        assert first.status == "created"
        assert second.status == "already_seeded"
        assert len(folders) == 1
        assert _seed_row(db, "local", "local")["state"] == "seeded"
        assert db.get_connection().execute("SELECT COUNT(*) FROM keywords").fetchone()[0] == 0
        assert db.get_connection().execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
    finally:
        db.close_connection()


def test_renamed_or_deleted_seed_is_never_recreated(tmp_path: Path):
    db = _db(tmp_path / "renamed.sqlite")
    folders = LocalNoteFolderRepository(db)
    try:
        seeded = initialize_agent_lessons_folder(
            db, scope_mode="local_only", profile_id="local", dataset_id="local"
        )
        current = folders.get_folder(seeded.folder_id)
        renamed = folders.rename_folder(
            seeded.folder_id, name="My lessons", expected_version=current.version
        )
        assert initialize_agent_lessons_folder(
            db, scope_mode="local_only", profile_id="local", dataset_id="local"
        ).status == "already_seeded"
        folders.soft_delete_folder(
            seeded.folder_id, expected_version=renamed.folder.version
        )
        assert initialize_agent_lessons_folder(
            db, scope_mode="local_only", profile_id="local", dataset_id="local"
        ).status == "already_seeded"
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM note_folders WHERE name = ? AND deleted = 0",
            (AGENT_LESSONS_FOLDER,),
        ).fetchone()[0] == 0
    finally:
        db.close_connection()


def test_exact_root_is_reused_but_case_variant_requires_review(tmp_path: Path):
    exact_db = _db(tmp_path / "exact.sqlite")
    variant_db = _db(tmp_path / "variant.sqlite")
    try:
        exact = LocalNoteFolderRepository(exact_db).create_folder(
            name=AGENT_LESSONS_FOLDER, parent_id=None
        )
        reused = initialize_agent_lessons_folder(
            exact_db, scope_mode="local_only", profile_id="local", dataset_id="local"
        )
        assert reused.status == "reused"
        assert reused.folder_id == exact.folder_id

        variant = LocalNoteFolderRepository(variant_db).create_folder(
            name="agent_lessons", parent_id=None
        )
        review = initialize_agent_lessons_folder(
            variant_db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=NotesOrganizationRepository(
                variant_db, server_profile_id=PROFILE
            ),
        )
        assert review.status == "adoption_review"
        row = variant_db.get_connection().execute(
            "SELECT local_object_id, remote_object_id, state FROM notes_organization_adoption_reviews"
        ).fetchone()
        assert dict(row) == {
            "local_object_id": variant.folder_id,
            "remote_object_id": None,
            "state": "open",
        }
        assert _seed_row(variant_db, PROFILE, DATASET)["state"] == "not_seeded"
    finally:
        exact_db.close_connection()
        variant_db.close_connection()


def test_synchronized_seed_records_creation_intent_and_state_together(tmp_path: Path):
    db = _db(tmp_path / "sync.sqlite")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    try:
        result = initialize_agent_lessons_folder(
            db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=repository,
        )
        folder = db.get_connection().execute(
            "SELECT * FROM note_folders WHERE id = ?", (result.folder_id,)
        ).fetchone()
        intent = db.get_connection().execute(
            "SELECT * FROM notes_organization_sync_intents WHERE domain = 'notes.folder'"
        ).fetchone()
        state = _seed_row(db, PROFILE, DATASET)
        assert result.status == "created"
        assert intent["object_id"] == folder["sync_id"]
        assert intent["source_version"] == 1
        assert intent["outbox_client_envelope_id"] is None
        assert state["folder_sync_id"] == folder["sync_id"]
        assert len(state["seed_fingerprint"]) == 64
    finally:
        db.close_connection()


def test_remote_exact_root_history_marks_seeded_before_duplicate_and_rename(tmp_path: Path):
    db = _db(tmp_path / "remote.sqlite")
    remote_id = str(uuid.uuid4())
    applier = SyncEnvelopeApplier(
        local_store=None,
        notes_organization_repository=NotesOrganizationRepository(
            db, server_profile_id=PROFILE
        ),
    )
    try:
        exact = _folder_envelope(remote_id, name=AGENT_LESSONS_FOLDER, revision=1, cursor=1)
        assert applier.apply(exact)["status"] == "applied"
        assert applier.apply(exact)["reason"] == "duplicate"
        renamed = _folder_envelope(remote_id, name="Renamed lessons", revision=2, cursor=2)
        assert applier.apply(renamed)["status"] == "applied"
        state = _seed_row(db, PROFILE, DATASET)
        assert state["state"] == "seeded"
        assert state["folder_sync_id"] == remote_id
    finally:
        db.close_connection()


def test_untouched_unpublished_two_device_seed_adopts_remote_identity(tmp_path: Path):
    db = _db(tmp_path / "race.sqlite")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    remote_id = str(uuid.uuid4())
    try:
        local = initialize_agent_lessons_folder(
            db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=repository,
        )
        local_sync_id = db.get_connection().execute(
            "SELECT sync_id FROM note_folders WHERE id = ?", (local.folder_id,)
        ).fetchone()[0]
        result = SyncEnvelopeApplier(
            local_store=None, notes_organization_repository=repository
        ).apply(_folder_envelope(remote_id, name=AGENT_LESSONS_FOLDER, revision=1, cursor=1))

        assert result["status"] == "applied"
        active = db.get_connection().execute(
            "SELECT sync_id FROM note_folders WHERE name = ? AND deleted = 0",
            (AGENT_LESSONS_FOLDER,),
        ).fetchall()
        assert [row["sync_id"] for row in active] == [remote_id]
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents WHERE object_id = ?",
            (local_sync_id,),
        ).fetchone()[0] == 0
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM notes_organization_adoption_reviews"
        ).fetchone()[0] == 0
        state = _seed_row(db, PROFILE, DATASET)
        assert state["folder_sync_id"] == remote_id
        assert state["seed_fingerprint"] == agent_lessons_seed_fingerprint(
            category="remote_history_upsert",
            profile_id=PROFILE,
            dataset_id=DATASET,
            folder_sync_id=remote_id,
        )
    finally:
        db.close_connection()


def test_stale_remote_history_records_evidence_without_retiring_local_seed(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path / "stale-history.sqlite")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    remote_id = str(uuid.uuid4())
    try:
        local = initialize_agent_lessons_folder(
            db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=repository,
        )
        local_sync_id = str(local.folder_sync_id)
        applier = SyncEnvelopeApplier(
            local_store=None, notes_organization_repository=repository
        )
        assert applier.apply(
            _folder_envelope(remote_id, name="Renamed lessons", revision=2, cursor=2)
        )["status"] == "applied"

        stale = applier.apply(
            _folder_envelope(
                remote_id, name=AGENT_LESSONS_FOLDER, revision=1, cursor=1
            )
        )

        assert stale == {"status": "noop", "reason": "stale"}
        local_folder = db.get_connection().execute(
            "SELECT deleted FROM note_folders WHERE sync_id = ?", (local_sync_id,)
        ).fetchone()
        assert local_folder["deleted"] == 0
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM notes_organization_sync_intents WHERE object_id = ?",
            (local_sync_id,),
        ).fetchone()[0] == 1
        review = db.get_connection().execute(
            "SELECT remote_object_id, state FROM notes_organization_adoption_reviews"
        ).fetchone()
        assert tuple(review) == (remote_id, "open")
        state = _seed_row(db, PROFILE, DATASET)
        assert state["folder_sync_id"] == remote_id
        assert state["seed_fingerprint"] == agent_lessons_seed_fingerprint(
            category="remote_history_upsert",
            profile_id=PROFILE,
            dataset_id=DATASET,
            folder_sync_id=remote_id,
        )
    finally:
        db.close_connection()


def test_copied_seed_race_requires_review(tmp_path: Path):
    db = _db(tmp_path / "copied.sqlite")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    remote_id = str(uuid.uuid4())
    try:
        initialize_agent_lessons_folder(
            db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=repository,
        )
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE notes_organization_sync_intents SET outbox_client_envelope_id = intent_id, copied_at = '2026-08-30T00:00:00Z'"
            )
        result = SyncEnvelopeApplier(
            local_store=None, notes_organization_repository=repository
        ).apply(_folder_envelope(remote_id, name=AGENT_LESSONS_FOLDER, revision=1, cursor=1))
        assert result["status"] == "conflict"
        assert db.get_connection().execute(
            "SELECT state FROM notes_organization_adoption_reviews"
        ).fetchone()["state"] == "open"
    finally:
        db.close_connection()


def test_local_case_variant_has_a_durable_review_instead_of_a_phantom_status(
    tmp_path: Path,
):
    db = _db(tmp_path / "local-variant.sqlite")
    try:
        variant = LocalNoteFolderRepository(db).create_folder(
            name="agent_lessons", parent_id=None
        )
        result = initialize_agent_lessons_folder(
            db, scope_mode="local_only", profile_id="local", dataset_id="local"
        )
        review = db.get_connection().execute(
            "SELECT local_object_id, remote_object_id, state FROM "
            "notes_organization_adoption_reviews WHERE server_profile_id = 'local' "
            "AND dataset_id = 'local'"
        ).fetchone()
        assert result.status == "adoption_review"
        assert dict(review) == {
            "local_object_id": variant.folder_id,
            "remote_object_id": None,
            "state": "open",
        }
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "unsafe_state", ["edited", "acknowledged", "used", "unrelated_intent"]
)
def test_non_pristine_seed_is_never_auto_retired(
    tmp_path: Path, unsafe_state: str
) -> None:
    db = _db(tmp_path / f"unsafe-{unsafe_state}.sqlite")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    try:
        seeded = initialize_agent_lessons_folder(
            db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=repository,
        )
        if unsafe_state == "edited":
            folder = LocalNoteFolderRepository(db).get_folder(seeded.folder_id)
            LocalNoteFolderRepository(db).rename_folder(
                seeded.folder_id,
                name="Edited Agent Lessons",
                expected_version=folder.version,
            )
        elif unsafe_state == "acknowledged":
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE notes_organization_sync_intents SET "
                    "outbox_client_envelope_id = intent_id, copied_at = ?, acknowledged_at = ?",
                    ("2026-08-30T00:00:00Z", "2026-08-30T00:00:01Z"),
                )
        elif unsafe_state == "used":
            note_id = db.add_note("Used", "Body", note_id=str(uuid.uuid4()))
            LocalNoteFolderRepository(db).attach_manual(
                folder_id=seeded.folder_id, note_id=note_id
            )
        else:
            unrelated = LocalNoteFolderRepository(db).create_folder(
                name="Unrelated", parent_id=None
            )
            with db.transaction() as cursor:
                unrelated_row = cursor.execute(
                    "SELECT sync_id, version FROM note_folders WHERE id = ?",
                    (unrelated.folder_id,),
                ).fetchone()
                repository.record_intent(
                    cursor,
                    profile=PROFILE,
                    dataset=DATASET,
                    domain="notes.folder",
                    object_id=str(unrelated_row["sync_id"]),
                    operation="upsert",
                    payload={"name": "Unrelated", "parent_sync_id": None},
                    source_version=int(unrelated_row["version"]),
                )

        remote_id = str(uuid.uuid4())
        SyncEnvelopeApplier(
            local_store=None, notes_organization_repository=repository
        ).apply(
            _folder_envelope(
                remote_id, name=AGENT_LESSONS_FOLDER, revision=1, cursor=1
            )
        )

        review = db.get_connection().execute(
            "SELECT remote_object_id, state FROM notes_organization_adoption_reviews"
        ).fetchone()
        assert tuple(review) == (remote_id, "open")
    finally:
        db.close_connection()


def test_differently_spelled_remote_seed_requires_review(tmp_path: Path) -> None:
    db = _db(tmp_path / "remote-variant.sqlite")
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    try:
        initialize_agent_lessons_folder(
            db,
            scope_mode="synchronized",
            profile_id=PROFILE,
            dataset_id=DATASET,
            organization_repository=repository,
        )
        result = SyncEnvelopeApplier(
            local_store=None, notes_organization_repository=repository
        ).apply(
            _folder_envelope(
                str(uuid.uuid4()), name="agent_lessons", revision=1, cursor=1
            )
        )
        assert result["status"] == "conflict"
        assert db.get_connection().execute(
            "SELECT state FROM notes_organization_adoption_reviews"
        ).fetchone()["state"] == "open"
    finally:
        db.close_connection()
