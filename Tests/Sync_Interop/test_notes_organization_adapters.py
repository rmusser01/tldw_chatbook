from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Sync_Interop.domain_adapters import (
    NotesOrganizationSyncAdapter,
)
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.notes_organization import (
    NOTES_ORGANIZATION_DOMAINS,
    organization_link_id,
)
from tldw_chatbook.Sync_Interop.restore_service import SyncRestoreService
from tldw_chatbook.tldw_api import SyncV2Envelope


DATASET = "dataset-a"


def _id(number: int) -> str:
    return str(uuid.UUID(f"00000000-0000-4000-8000-{number:012d}"))


def _envelope(
    domain: str,
    object_id: str,
    payload: dict[str, object],
    *,
    operation: str = "upsert",
    revision: int = 1,
    cursor: int = 1,
    routing_metadata: dict[str, object] | None = None,
    base_server_cursor: int | None = None,
    base_object_revision: int | None = None,
    base_object_hash: str | None = None,
) -> SyncV2Envelope:
    content = json.dumps(
        {"operation": operation, "payload": payload, "revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SyncV2Envelope(
        client_envelope_id=f"remote:{domain}:{object_id}:{revision}",
        dataset_id=DATASET,
        device_id="remote-device",
        domain=domain,
        object_id=object_id,
        operation=operation,
        schema_version=1,
        object_revision=revision,
        server_cursor=cursor,
        payload=payload,
        payload_hash=hashlib.sha256(content).hexdigest(),
        encryption_policy="server_trusted_v1",
        routing_metadata=routing_metadata or {},
        base_server_cursor=base_server_cursor,
        base_object_revision=base_object_revision,
        base_object_hash=base_object_hash,
    )


@pytest.fixture
def organization_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="adapter-tests")
    try:
        yield db
    finally:
        db.close_connection()


def _applier(
    db: CharactersRAGDB,
    *,
    repository: NotesOrganizationRepository | None = None,
) -> SyncEnvelopeApplier:
    return SyncEnvelopeApplier(
        local_store=None,
        notes_organization_repository=(
            repository or NotesOrganizationRepository(db, server_profile_id="server-a")
        ),
    )


def _seed_note(db: CharactersRAGDB, note_id: str) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes(id, title, content, client_id, version) "
            "VALUES (?, 'Note', 'private body', 'adapter-tests', 1)",
            (note_id,),
        )


def test_applier_registers_one_adapter_family_for_all_six_domains(
    organization_db: CharactersRAGDB,
) -> None:
    applier = _applier(organization_db)

    adapters = [applier._adapters[domain] for domain in NOTES_ORGANIZATION_DOMAINS]

    assert all(
        isinstance(adapter, NotesOrganizationSyncAdapter) for adapter in adapters
    )
    assert len({id(adapter) for adapter in adapters}) == 1


def test_applier_projects_every_notes_organization_domain(
    organization_db: CharactersRAGDB,
) -> None:
    keyword_id, collection_id, folder_id, note_id = (_id(i) for i in range(1, 5))
    _seed_note(organization_db, note_id)
    keyword_link = {
        "subject_type": "note",
        "subject_id": note_id,
        "keyword_sync_id": keyword_id,
    }
    collection_link = {
        "collection_sync_id": collection_id,
        "keyword_sync_id": keyword_id,
    }
    folder_link = {"note_id": note_id, "folder_sync_id": folder_id}
    envelopes = (
        _envelope("notes.keyword", keyword_id, {"keyword": "Research"}, cursor=1),
        _envelope(
            "notes.keyword_collection",
            collection_id,
            {"name": "Sources", "parent_sync_id": None},
            cursor=2,
        ),
        _envelope(
            "notes.folder",
            folder_id,
            {"name": "Agent_Lessons", "parent_sync_id": None},
            cursor=3,
        ),
        _envelope(
            "notes.keyword_link",
            organization_link_id("notes.keyword_link", list(keyword_link.values())),
            keyword_link,
            cursor=4,
        ),
        _envelope(
            "notes.keyword_collection_link",
            organization_link_id(
                "notes.keyword_collection_link", list(collection_link.values())
            ),
            collection_link,
            cursor=5,
        ),
        _envelope(
            "notes.folder_link",
            organization_link_id("notes.folder_link", list(folder_link.values())),
            folder_link,
            cursor=6,
        ),
    )

    results = [_applier(organization_db).apply(envelope) for envelope in envelopes]

    assert [result["status"] for result in results] == ["applied"] * 6
    heads = (
        organization_db.get_connection()
        .execute(
            "SELECT domain, apply_state FROM notes_organization_heads ORDER BY domain"
        )
        .fetchall()
    )
    assert {row["domain"] for row in heads} == set(NOTES_ORGANIZATION_DOMAINS)
    assert {row["apply_state"] for row in heads} == {"applied"}


def test_applier_reports_duplicate_and_stale_heads_as_noops(
    organization_db: CharactersRAGDB,
) -> None:
    object_id = _id(10)
    applier = _applier(organization_db)
    current = _envelope(
        "notes.keyword", object_id, {"keyword": "Current"}, revision=2, cursor=2
    )
    stale = _envelope(
        "notes.keyword", object_id, {"keyword": "Stale"}, revision=1, cursor=1
    )

    assert applier.apply(current)["status"] == "applied"
    assert applier.apply(current) == {"status": "noop", "reason": "duplicate"}
    assert applier.apply(stale) == {"status": "noop", "reason": "stale"}


def test_exact_agent_lessons_root_history_survives_later_tombstone(
    organization_db: CharactersRAGDB,
) -> None:
    object_id = _id(12)
    applier = _applier(organization_db)

    assert applier.apply(
        _envelope(
            "notes.folder",
            object_id,
            {"name": "Agent_Lessons", "parent_sync_id": None},
            cursor=1,
        )
    )["status"] == "applied"
    assert applier.apply(
        _envelope(
            "notes.folder",
            object_id,
            {},
            operation="tombstone",
            revision=2,
            cursor=2,
        )
    )["status"] == "applied"

    state = organization_db.get_connection().execute(
        "SELECT state, folder_sync_id FROM agent_lessons_seed_state WHERE "
        "profile_id = 'server-a' AND dataset_id = ?",
        (DATASET,),
    ).fetchone()
    assert tuple(state) == ("seeded", object_id)


@pytest.mark.parametrize(
    "envelope",
    [
        _envelope("notes.keyword", "not-a-uuid", {"keyword": "Bad identity"}),
        _envelope("notes.keyword", _id(20), {"keyword": "   "}),
        _envelope("notes.folder", _id(21), {"name": "Folder", "unexpected": True}),
    ],
)
def test_applier_rejects_invalid_identity_and_payload_without_exposing_content(
    organization_db: CharactersRAGDB,
    envelope: SyncV2Envelope,
) -> None:
    result = _applier(organization_db).apply(envelope)

    assert result["status"] == "rejected"
    assert result["error_code"].startswith("notes_organization_")
    assert "Bad identity" not in json.dumps(result)


def test_applier_rejects_missing_repository_dependency() -> None:
    envelope = _envelope("notes.keyword", _id(30), {"keyword": "Research"})

    result = SyncEnvelopeApplier(local_store=None).apply(envelope)

    assert result == {
        "status": "rejected",
        "error_code": "notes_organization_repository_unavailable",
    }


def test_blocked_dependency_is_durable_and_replayable(
    organization_db: CharactersRAGDB,
) -> None:
    note_id, folder_id = _id(40), _id(41)
    payload = {"note_id": note_id, "folder_sync_id": folder_id}
    link_id = organization_link_id("notes.folder_link", list(payload.values()))
    envelope = _envelope("notes.folder_link", link_id, payload)
    applier = _applier(organization_db)

    blocked = applier.apply(envelope)
    head = (
        organization_db.get_connection()
        .execute(
            "SELECT apply_state FROM notes_organization_heads "
            "WHERE domain = 'notes.folder_link' AND object_id = ?",
            (link_id,),
        )
        .fetchone()
    )

    assert blocked["status"] == "conflict"
    assert blocked["conflict"]["conflict_type"] == "missing_dependency"
    assert dict(head) == {"apply_state": "blocked"}

    _seed_note(organization_db, note_id)
    assert (
        applier.apply(
            _envelope(
                "notes.folder",
                folder_id,
                {"name": "Agent_Lessons", "parent_sync_id": None},
                cursor=2,
            )
        )["status"]
        == "applied"
    )
    assert applier.apply(envelope)["status"] == "applied"


def test_hierarchy_cycle_is_a_safe_durable_conflict(
    organization_db: CharactersRAGDB,
) -> None:
    parent_id, child_id = _id(50), _id(51)
    applier = _applier(organization_db)
    assert (
        applier.apply(
            _envelope(
                "notes.folder",
                parent_id,
                {"name": "Parent", "parent_sync_id": None},
                cursor=1,
            )
        )["status"]
        == "applied"
    )
    assert (
        applier.apply(
            _envelope(
                "notes.folder",
                child_id,
                {"name": "Child", "parent_sync_id": parent_id},
                cursor=2,
            )
        )["status"]
        == "applied"
    )

    result = applier.apply(
        _envelope(
            "notes.folder",
            parent_id,
            {"name": "Parent", "parent_sync_id": child_id},
            revision=2,
            cursor=3,
        )
    )

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "hierarchy_cycle"
    head = (
        organization_db.get_connection()
        .execute(
            "SELECT apply_state FROM notes_organization_heads "
            "WHERE domain = 'notes.folder' AND object_id = ?",
            (parent_id,),
        )
        .fetchone()
    )
    assert dict(head) == {"apply_state": "blocked"}


def test_hierarchy_collision_is_a_safe_durable_adoption_conflict(
    organization_db: CharactersRAGDB,
) -> None:
    with organization_db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folders(id, parent_id, name, normalized_name, path, "
            "normalized_path, version, deleted, created_at, modified_at) VALUES "
            "('local-folder', NULL, 'Agent_Lessons', 'agent_lessons', "
            "'/Agent_Lessons', '/agent_lessons', 1, 0, "
            "'2026-08-29T00:00:00Z', '2026-08-29T00:00:00Z')"
        )
    remote_id = _id(55)

    result = _applier(organization_db).apply(
        _envelope(
            "notes.folder",
            remote_id,
            {"name": "Agent_Lessons", "parent_sync_id": None},
        )
    )

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "local_representation_collision"
    review = (
        organization_db.get_connection()
        .execute(
            "SELECT state, remote_object_id FROM notes_organization_adoption_reviews "
            "WHERE domain = 'notes.folder'"
        )
        .fetchone()
    )
    assert dict(review) == {"state": "open", "remote_object_id": remote_id}


def test_resource_restore_requires_explicit_restore_flow(
    organization_db: CharactersRAGDB,
) -> None:
    keyword_id = _id(60)
    ordinary = _applier(organization_db)
    upsert = _envelope("notes.keyword", keyword_id, {"keyword": "Research"})
    tombstone = _envelope(
        "notes.keyword", keyword_id, {}, operation="tombstone", revision=2, cursor=2
    )
    restore = _envelope(
        "notes.keyword",
        keyword_id,
        {"keyword": "Research"},
        revision=3,
        cursor=3,
    )

    assert ordinary.apply(upsert)["status"] == "applied"
    assert ordinary.apply(tombstone)["status"] == "applied"
    blocked = ordinary.apply(restore)
    assert blocked["status"] == "conflict"
    assert blocked["conflict"]["conflict_type"] == "restore_intent_required"

    truthy_restore = restore.model_copy(
        update={"routing_metadata": {"restore_intent": "true"}}
    )
    blocked_truthy = ordinary.apply(truthy_restore)
    assert blocked_truthy["status"] == "conflict"
    assert blocked_truthy["conflict"]["conflict_type"] == "restore_intent_required"

    literal_restore = restore.model_copy(
        update={
            "routing_metadata": {"restore_intent": True},
            "base_object_revision": tombstone.object_revision,
            "base_object_hash": tombstone.payload_hash,
        }
    )
    assert ordinary.apply(literal_restore)["status"] == "applied"

    row = (
        organization_db.get_connection()
        .execute("SELECT deleted FROM keywords WHERE sync_id = ?", (keyword_id,))
        .fetchone()
    )
    assert row["deleted"] == 0


def test_literal_restore_requires_the_exact_current_tombstone_base(
    organization_db: CharactersRAGDB,
) -> None:
    ordinary = _applier(organization_db)
    active_id, missing_id = _id(61), _id(62)
    active = _envelope("notes.keyword", active_id, {"keyword": "Active"})
    tombstone = _envelope(
        "notes.keyword", active_id, {}, operation="tombstone", revision=2, cursor=2
    )
    assert ordinary.apply(active)["status"] == "applied"

    spurious_active_restore = _envelope(
        "notes.keyword",
        active_id,
        {"keyword": "Active"},
        revision=2,
        cursor=2,
        routing_metadata={"restore_intent": True},
        base_object_revision=active.object_revision,
        base_object_hash=active.payload_hash,
    )
    assert ordinary.apply(spurious_active_restore)["status"] == "conflict"

    spurious_missing_restore = _envelope(
        "notes.keyword",
        missing_id,
        {"keyword": "Missing"},
        routing_metadata={"restore_intent": True},
        base_object_revision=1,
        base_object_hash="a" * 64,
    )
    assert ordinary.apply(spurious_missing_restore)["status"] == "conflict"

    assert ordinary.apply(tombstone)["status"] == "applied"
    stale_restore = _envelope(
        "notes.keyword",
        active_id,
        {"keyword": "Restored"},
        revision=3,
        cursor=3,
        routing_metadata={"restore_intent": True},
        base_object_revision=1,
        base_object_hash=active.payload_hash,
    )
    assert ordinary.apply(stale_restore)["status"] == "conflict"

    exact_restore_without_cursor = stale_restore.model_copy(
        update={
            "base_object_revision": tombstone.object_revision,
            "base_object_hash": tombstone.payload_hash,
        }
    )
    assert ordinary.apply(exact_restore_without_cursor)["status"] == "applied"


def test_link_restore_cannot_bypass_restore_intent(
    organization_db: CharactersRAGDB,
) -> None:
    note_id, folder_id = _id(70), _id(71)
    _seed_note(organization_db, note_id)
    ordinary = _applier(organization_db)
    assert (
        ordinary.apply(
            _envelope(
                "notes.folder",
                folder_id,
                {"name": "Agent_Lessons", "parent_sync_id": None},
            )
        )["status"]
        == "applied"
    )
    payload = {"note_id": note_id, "folder_sync_id": folder_id}
    link_id = organization_link_id("notes.folder_link", list(payload.values()))
    assert (
        ordinary.apply(_envelope("notes.folder_link", link_id, payload, cursor=2))[
            "status"
        ]
        == "applied"
    )
    assert (
        ordinary.apply(
            _envelope(
                "notes.folder_link",
                link_id,
                payload,
                operation="tombstone",
                revision=2,
                cursor=3,
            )
        )["status"]
        == "applied"
    )

    blocked = ordinary.apply(
        _envelope("notes.folder_link", link_id, payload, revision=3, cursor=4)
    )

    assert blocked["status"] == "conflict"
    assert blocked["conflict"]["conflict_type"] == "restore_intent_required"


@pytest.mark.parametrize("failure_stage", ["projection", "head"])
def test_projection_or_head_failure_rolls_back_without_leaking_error(
    organization_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repository = NotesOrganizationRepository(
        organization_db, server_profile_id="server-a"
    )

    if failure_stage == "projection":
        original_materialize = repository._materialize

        def fail_projection(*args, **kwargs):
            original_materialize(*args, **kwargs)
            raise RuntimeError("SECRET SQL failure details")

        monkeypatch.setattr(repository, "_materialize", fail_projection)
    else:

        def fail_head(*args, **kwargs):
            raise RuntimeError("SECRET SQL failure details")

        monkeypatch.setattr(repository, "_write_head", fail_head)
    object_id = _id(80)

    result = _applier(organization_db, repository=repository).apply(
        _envelope("notes.keyword", object_id, {"keyword": "Rollback"})
    )

    assert result == {
        "status": "rejected",
        "error_code": "notes_organization_apply_failed",
    }
    assert "SECRET" not in json.dumps(result)
    assert (
        organization_db.get_connection()
        .execute("SELECT 1 FROM keywords WHERE sync_id = ?", (object_id,))
        .fetchone()
        is None
    )
    assert (
        organization_db.get_connection()
        .execute(
            "SELECT 1 FROM notes_organization_heads WHERE object_id = ?", (object_id,)
        )
        .fetchone()
        is None
    )


def test_adapter_acquires_write_lock_before_read_then_write(tmp_path: Path) -> None:
    db_path = tmp_path / "contended-notes.sqlite"
    primary = CharactersRAGDB(db_path, client_id="primary")
    writer_ready = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[Exception] = []

    def hold_competing_write_lock() -> None:
        competing = CharactersRAGDB(db_path, client_id="competing")
        try:
            with competing.transaction(immediate=True):
                writer_ready.set()
                if not release_writer.wait(2):
                    raise AssertionError("competing writer was not released")
        except Exception as exc:  # pragma: no cover - asserted below
            writer_errors.append(exc)
        finally:
            competing.close_connection()

    writer = threading.Thread(target=hold_competing_write_lock)
    writer.start()
    assert writer_ready.wait(2)
    timer = threading.Timer(0.05, release_writer.set)
    timer.start()
    object_id = _id(81)
    try:
        result = _applier(primary).apply(
            _envelope("notes.keyword", object_id, {"keyword": "Contended"})
        )
    finally:
        release_writer.set()
        timer.cancel()
        writer.join(2)

    try:
        assert writer_errors == []
        assert not writer.is_alive()
        assert result == {"status": "applied"}
        assert (
            primary.get_connection()
            .execute("SELECT 1 FROM keywords WHERE sync_id = ?", (object_id,))
            .fetchone()
            is not None
        )
        assert (
            primary.get_connection()
            .execute(
                "SELECT 1 FROM notes_organization_heads WHERE object_id = ?",
                (object_id,),
            )
            .fetchone()
            is not None
        )
    finally:
        primary.close_connection()


@pytest.mark.asyncio
async def test_restore_service_alone_passes_explicit_restore_intent(
    organization_db: CharactersRAGDB,
) -> None:
    keyword_id = _id(90)
    ordinary = _applier(organization_db)
    assert (
        ordinary.apply(_envelope("notes.keyword", keyword_id, {"keyword": "Research"}))[
            "status"
        ]
        == "applied"
    )
    tombstone = _envelope(
        "notes.keyword",
        keyword_id,
        {},
        operation="tombstone",
        revision=2,
        cursor=2,
    )
    assert ordinary.apply(tombstone)["status"] == "applied"
    restore_envelope = _envelope(
        "notes.keyword",
        keyword_id,
        {"keyword": "Research"},
        revision=3,
        cursor=3,
        routing_metadata={"restore_intent": True},
        base_object_revision=tombstone.object_revision,
        base_object_hash=tombstone.payload_hash,
    )

    class RestoreServer:
        async def pull_v2_envelopes(self, **kwargs):
            return {
                "dataset_id": DATASET,
                "envelopes": [restore_envelope.model_dump(mode="json")],
                "next_cursor": "3",
                "has_more": False,
            }

    service = SyncRestoreService(
        server_service=RestoreServer(),
        local_store=None,
        dataset_keys={DATASET: generate_dataset_key()},
        notes_organization_repository=NotesOrganizationRepository(organization_db),
    )

    result = await service.restore_selection(
        server_profile_id="server-a",
        dataset_id=DATASET,
        device_id="restore-device",
        domains=["notes.keyword"],
    )

    assert result["applied"] == 1
    assert result["conflicts"] == []
    head = (
        organization_db.get_connection()
        .execute(
            "SELECT server_profile_id FROM notes_organization_heads "
            "WHERE dataset_id = ? AND object_id = ?",
            (DATASET, keyword_id),
        )
        .fetchone()
    )
    assert head["server_profile_id"] == "server-a"


@pytest.mark.asyncio
async def test_organization_only_restore_does_not_require_dataset_key(
    organization_db: CharactersRAGDB,
) -> None:
    envelope = _envelope(
        "notes.keyword",
        _id(92),
        {"keyword": "Clear restore"},
    )

    class RestoreServer:
        async def pull_v2_envelopes(self, **kwargs):
            return {
                "dataset_id": DATASET,
                "envelopes": [envelope.model_dump(mode="json")],
                "next_cursor": "1",
                "has_more": False,
            }

    service = SyncRestoreService(
        server_service=RestoreServer(),
        local_store=None,
        dataset_keys={},
        notes_organization_repository=NotesOrganizationRepository(organization_db),
    )

    result = await service.restore_selection(
        server_profile_id="server-b",
        dataset_id=DATASET,
        device_id="restore-device",
        domains=["notes.keyword"],
    )

    assert result["applied"] == 1
    head = (
        organization_db.get_connection()
        .execute(
            "SELECT server_profile_id FROM notes_organization_heads "
            "WHERE object_id = ?",
            (envelope.object_id,),
        )
        .fetchone()
    )
    assert head["server_profile_id"] == "server-b"


def test_restore_service_preserves_empty_shared_dataset_key_cache() -> None:
    shared_dataset_keys: dict[str, bytes] = {}

    service = SyncRestoreService(
        server_service=object(),
        local_store=None,
        dataset_keys=shared_dataset_keys,
    )

    assert service.dataset_keys is shared_dataset_keys


@pytest.mark.asyncio
async def test_restore_service_observes_dataset_key_added_after_construction(
    organization_db: CharactersRAGDB,
) -> None:
    envelope = _envelope(
        "notes.keyword",
        _id(91),
        {"keyword": "Late key"},
    )

    class RestoreServer:
        async def pull_v2_envelopes(self, **kwargs):
            return {
                "dataset_id": DATASET,
                "envelopes": [envelope.model_dump(mode="json")],
                "next_cursor": "1",
                "has_more": False,
            }

    shared_dataset_keys: dict[str, bytes] = {}
    service = SyncRestoreService(
        server_service=RestoreServer(),
        local_store=None,
        dataset_keys=shared_dataset_keys,
        notes_organization_repository=NotesOrganizationRepository(
            organization_db, server_profile_id="server-a"
        ),
    )
    shared_dataset_keys[DATASET] = generate_dataset_key()

    result = await service.restore_selection(
        server_profile_id="server-a",
        dataset_id=DATASET,
        device_id="restore-device",
        domains=["notes.keyword"],
    )

    assert result["applied"] == 1
