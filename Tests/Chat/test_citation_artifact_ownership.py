from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json

import pytest
from pydantic import ValidationError

from Tests.Chat.test_citation_payload_lifecycle import _mark_owner_deleted, _policy
from Tests.Chat.test_citation_trace_repository import (
    _identity,
    _persist,
    _repository,
)
from tldw_chatbook.Chat.citation_artifact_ownership import (
    ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES,
    ArtifactBackendMode,
    ArtifactOwnerBinding,
    ArtifactOwnerOperation,
    ArtifactOwnerOperationKind,
    ArtifactOwnerOutboxState,
    CitationArtifactOwnershipCoordinator,
)
from tldw_chatbook.Chat.citation_payload_lifecycle import CitationPayloadLifecycle
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)
from tldw_chatbook.Chatbooks import LocalChatbookService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


NOW = datetime(2026, 7, 24, 18, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        tmp_path / "citation-artifact-ownership.sqlite",
        client_id="citation-artifact-ownership-test",
    )
    yield database
    database.close_connection()


@pytest.fixture
def ownership(
    db: CharactersRAGDB,
    tmp_path,
) -> tuple[
    CitationTraceRepository,
    LocalChatbookService,
    CitationArtifactOwnershipCoordinator,
]:
    repository = _repository(db)
    _persist(db, repository)
    service = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "chatbooks.json",
    )
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=repository,
    )
    service.set_citation_ownership_coordinator(coordinator)
    return repository, service, coordinator


def _owner_request(repository: CitationTraceRepository):
    request = repository.get_artifact_owner_request(
        message_id="message-1",
        message_revision=1,
        current_body="Answer [S1].",
    )
    assert request is not None
    return request


def _outbox(service: LocalChatbookService) -> list[ArtifactOwnerOperation]:
    return service.list_provenance_outbox(limit=ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES)


def test_current_json_backend_is_cross_store_and_shared_db_requires_real_contract(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    service = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "chatbooks.json",
    )
    assert service.artifact_backend_mode is ArtifactBackendMode.CROSS_STORE

    class FakeSharedDatabaseStore:
        artifact_backend_mode = ArtifactBackendMode.SHARED_DATABASE

        def apply_shared_database_owner_mutation(self) -> None:
            """A callable alone cannot prove a real FK/shared transaction."""

    with pytest.raises(ValueError, match="shared_database_owner_contract_required"):
        CitationArtifactOwnershipCoordinator(
            artifact_store=FakeSharedDatabaseStore(),
            trace_repository=_repository(db),
        )


def test_owner_models_are_strict_frozen_bounded_and_reject_boolean_revisions() -> None:
    operation = ArtifactOwnerOperation(
        operation_id="operation-1",
        operation_kind=ArtifactOwnerOperationKind.LINK,
        binding=ArtifactOwnerBinding(
            profile_id="profile-1",
            artifact_store_id="local-chatbooks-v1",
            artifact_id="artifact-1",
            artifact_revision=1,
            trace_id="trace-1",
            lease_id="lease-1",
            binding_id="binding-1",
        ),
        state=ArtifactOwnerOutboxState.PENDING,
        created_at=NOW,
    )
    with pytest.raises(ValidationError, match="frozen"):
        operation.state = ArtifactOwnerOutboxState.ACKNOWLEDGED  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ArtifactOwnerBinding(
            profile_id="profile-1",
            artifact_store_id="local-chatbooks-v1",
            artifact_id="artifact-1",
            artifact_revision=True,
            trace_id="trace-1",
            lease_id="lease-1",
            binding_id="binding-1",
        )
    corrupted = operation.model_copy(update={"operation_id": "x" * 257})
    with pytest.raises(ValidationError):
        ArtifactOwnerOperation.model_validate(
            corrupted.model_dump(mode="python"),
            strict=True,
        )
    unsafe_error = operation.model_copy(update={"error_code": "private token value"})
    with pytest.raises(ValidationError):
        ArtifactOwnerOperation.model_validate(
            unsafe_error.model_dump(mode="python"),
            strict=True,
        )


@pytest.mark.asyncio
async def test_artifact_create_and_link_outbox_are_one_atomic_registry_write(
    ownership,
    monkeypatch,
) -> None:
    repository, service, _coordinator = ownership
    saved_payloads: list[dict] = []
    original_save = service._save_registry

    def capture(payload: dict) -> None:
        saved_payloads.append(json.loads(json.dumps(payload)))
        original_save(payload)

    monkeypatch.setattr(service, "_save_registry", capture)
    created = await service.create_chatbook(
        name="Grounded answer",
        metadata={
            "artifact_source": "console",
            "artifact_kind": "assistant-response",
            "content": "Answer [S1].",
        },
        provenance_owner_request=_owner_request(repository),
    )

    assert len(saved_payloads) == 1
    assert saved_payloads[0]["records"][0]["chatbook_id"] == created["chatbook_id"]
    operation = ArtifactOwnerOperation.model_validate_json(
        json.dumps(saved_payloads[0]["provenance_outbox"][0]),
        strict=True,
    )
    assert operation.operation_kind is ArtifactOwnerOperationKind.LINK
    assert operation.state is ArtifactOwnerOutboxState.PENDING
    assert operation.binding.artifact_id == str(created["chatbook_id"])
    assert created["artifact_revision"] == 1
    serialized = json.dumps(saved_payloads[0])
    for governed_value in (
        "private exact submitted evidence",
        "private source title",
        "private-document",
        "private-item",
        "content-hmac",
        "comparison-hmac",
    ):
        assert governed_value not in serialized


@pytest.mark.asyncio
async def test_old_registry_without_outbox_loads_and_new_outbox_is_bounded(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    registry_path = tmp_path / "chatbooks.json"
    registry_path.write_text(
        json.dumps({"next_id": 1, "records": []}),
        encoding="utf-8",
    )
    service = LocalChatbookService(db_paths={}, registry_path=registry_path)
    assert _outbox(service) == []

    oversized = {
        "next_id": 1,
        "records": [],
        "provenance_outbox": [{}] * (ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES + 1),
    }
    registry_path.write_text(json.dumps(oversized), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance_outbox"):
        service.list_provenance_outbox(limit=ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES)


@pytest.mark.asyncio
async def test_delete_records_distinct_stable_unlink_for_the_same_lease(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    link = _outbox(service)[0]

    assert await service.delete_chatbook(created["chatbook_id"]) is True

    operations = _outbox(service)
    assert [item.operation_kind for item in operations] == [
        ArtifactOwnerOperationKind.LINK,
        ArtifactOwnerOperationKind.UNLINK,
    ]
    unlink = operations[1]
    assert unlink.operation_id != link.operation_id
    assert unlink.binding == link.binding

    # Replaying preparation for the same stable binding cannot allocate a
    # different operation identity.
    assert (
        ownership[2].prepare_unlink_operation(link.binding).operation_id
        == unlink.operation_id
    )


@pytest.mark.asyncio
async def test_registry_binding_cannot_be_moved_to_a_different_artifact(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    registry["records"][0]["id"] = "2"
    registry["records"][0]["chatbook_id"] = 2
    service.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance registry"):
        await service.delete_chatbook(2)


@pytest.mark.asyncio
async def test_replacement_save_allocates_next_revision_and_one_link_unlink_pair(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    replaced = await service.update_chatbook(
        created["chatbook_id"],
        name="Grounded answer v2",
        provenance_owner_request=_owner_request(repository),
    )

    operations = _outbox(service)
    assert replaced["artifact_revision"] == 2
    assert [
        (item.binding.artifact_revision, item.operation_kind) for item in operations
    ] == [
        (1, ArtifactOwnerOperationKind.LINK),
        (1, ArtifactOwnerOperationKind.UNLINK),
        (2, ArtifactOwnerOperationKind.LINK),
    ]
    assert len({item.operation_id for item in operations}) == 3


@pytest.mark.asyncio
async def test_reconcile_link_and_unlink_use_one_lease_and_durable_receipts(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    link = _outbox(service)[0]
    assert coordinator.reconcile_pending(limit=1).completed == 1
    assert _outbox(service) == []

    row = (
        repository.db.get_connection()
        .execute(
            """
        SELECT state, lease_id FROM rag_artifact_owner_leases
        WHERE profile_id = ? AND artifact_id = ?
        """,
            (_identity(repository.db).profile_id, str(created["chatbook_id"])),
        )
        .fetchone()
    )
    assert tuple(row) == ("live", link.binding.lease_id)

    await service.delete_chatbook(created["chatbook_id"])
    unlink = _outbox(service)[0]
    assert coordinator.reconcile_pending(limit=1).completed == 1
    assert _outbox(service) == []

    connection = repository.db.get_connection()
    lease = connection.execute(
        """
        SELECT state, lease_id FROM rag_artifact_owner_leases
        WHERE profile_id = ? AND artifact_id = ?
        """,
        (_identity(repository.db).profile_id, str(created["chatbook_id"])),
    ).fetchone()
    receipts = connection.execute(
        """
        SELECT operation_id, operation_kind, state
        FROM rag_artifact_owner_operations
        ORDER BY operation_kind
        """
    ).fetchall()
    assert tuple(lease) == ("released", link.binding.lease_id)
    assert {tuple(row) for row in receipts} == {
        (link.operation_id, "link", "acknowledged"),
        (unlink.operation_id, "unlink", "acknowledged"),
    }

    # Replaying after registry cleanup remains idempotent and cannot resurrect.
    repository.apply_artifact_owner_operation(link)
    repository.acknowledge_artifact_owner_operation(
        link.model_copy(
            update={
                "state": ArtifactOwnerOutboxState.ACKNOWLEDGED,
                "acknowledged_at": NOW,
            }
        )
    )
    assert (
        connection.execute("SELECT state FROM rag_artifact_owner_leases").fetchone()[0]
        == "released"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["apply", "registry_ack", "release"])
async def test_restart_recovers_each_unlink_handshake_phase(
    ownership,
    monkeypatch,
    failure_phase: str,
) -> None:
    repository, service, coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    assert coordinator.reconcile_pending(limit=1).completed == 1
    await service.delete_chatbook(created["chatbook_id"])

    if failure_phase == "apply":
        monkeypatch.setattr(
            repository,
            "apply_artifact_owner_operation",
            lambda operation: (_ for _ in ()).throw(RuntimeError("phase_apply")),
        )
    elif failure_phase == "registry_ack":
        monkeypatch.setattr(
            service,
            "mark_provenance_operation_acknowledged",
            lambda operation_id: (_ for _ in ()).throw(RuntimeError("phase_ack")),
        )
    else:
        monkeypatch.setattr(
            repository,
            "acknowledge_artifact_owner_operation",
            lambda operation: (_ for _ in ()).throw(RuntimeError("phase_release")),
        )

    failed_operation_id = _outbox(service)[0].operation_id
    result = coordinator.reconcile_pending(limit=1)
    assert result.failed == 1
    assert result.operation_ids == (failed_operation_id,)
    assert _outbox(service)[0].error_code == "artifact_reconciliation_failed"
    monkeypatch.undo()
    restarted = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=repository,
    )
    service.set_citation_ownership_coordinator(restarted)
    assert restarted.reconcile_pending(limit=1).completed == 1
    assert _outbox(service) == []
    assert (
        repository.db.get_connection()
        .execute("SELECT state FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == "released"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["apply", "registry_ack", "finalize"])
async def test_restart_recovers_each_link_handshake_phase(
    ownership,
    monkeypatch,
    failure_phase: str,
) -> None:
    repository, service, coordinator = ownership
    await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )

    if failure_phase == "apply":
        monkeypatch.setattr(
            repository,
            "apply_artifact_owner_operation",
            lambda operation: (_ for _ in ()).throw(RuntimeError("phase_apply")),
        )
    elif failure_phase == "registry_ack":
        monkeypatch.setattr(
            service,
            "mark_provenance_operation_acknowledged",
            lambda operation_id: (_ for _ in ()).throw(RuntimeError("phase_ack")),
        )
    else:
        monkeypatch.setattr(
            repository,
            "acknowledge_artifact_owner_operation",
            lambda operation: (_ for _ in ()).throw(RuntimeError("phase_finalize")),
        )

    failed_operation_id = _outbox(service)[0].operation_id
    result = coordinator.reconcile_pending(limit=1)
    assert result.failed == 1
    assert result.operation_ids == (failed_operation_id,)
    assert _outbox(service)[0].error_code == "artifact_reconciliation_failed"
    monkeypatch.undo()
    restarted = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=repository,
    )
    service.set_citation_ownership_coordinator(restarted)
    assert restarted.reconcile_pending(limit=1).completed == 1
    assert _outbox(service) == []
    assert (
        repository.db.get_connection()
        .execute("SELECT state FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == "live"
    )


def test_overlapping_registry_link_creates_do_not_drop_outbox_entries(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    owner_request = _owner_request(repository)

    def create_one(index: int) -> int:
        result = asyncio.run(
            service.create_chatbook(
                name=f"Grounded answer {index}",
                provenance_owner_request=owner_request,
            )
        )
        return int(result["chatbook_id"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        artifact_ids = list(executor.map(create_one, range(16)))

    assert len(set(artifact_ids)) == 16
    operations = _outbox(service)
    assert len(operations) == 16
    assert {item.binding.artifact_id for item in operations} == {
        str(artifact_id) for artifact_id in artifact_ids
    }


@pytest.mark.asyncio
async def test_pending_outbox_live_lease_and_unresolved_unlink_block_collection(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    owner_request = _owner_request(repository)
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=owner_request,
    )
    _mark_owner_deleted(repository.db)
    lifecycle = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
        artifact_barrier_provider=coordinator.collection_barriers,
    )

    assert lifecycle.collect(now=NOW).traces_collected == 0
    link = _outbox(service)[0]
    repository.apply_artifact_owner_operation(link)
    service.mark_provenance_operation_acknowledged(link.operation_id)
    assert lifecycle.collect(now=NOW).traces_collected == 0
    repository.acknowledge_artifact_owner_operation(_outbox(service)[0])
    service.prune_provenance_operation(link.operation_id)

    await service.delete_chatbook(created["chatbook_id"])
    unlink = _outbox(service)[0]
    repository.apply_artifact_owner_operation(unlink)
    service.mark_provenance_operation_acknowledged(unlink.operation_id)
    assert lifecycle.collect(now=NOW).traces_collected == 0


def test_corrupt_registry_fails_closed_for_reconciliation_and_collection(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    _mark_owner_deleted(repository.db)
    service.registry_path.write_text("{not-json", encoding="utf-8")

    result = coordinator.reconcile_pending(limit=1)
    assert result.failed == 1
    assert result.reason_codes == ("artifact_registry_unavailable",)
    lifecycle = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
        artifact_barrier_provider=coordinator.collection_barriers,
    )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="artifact_registry_unavailable",
    ):
        lifecycle.collect(now=NOW)


@pytest.mark.asyncio
async def test_reconciliation_never_persists_untrusted_exception_text(
    ownership,
    monkeypatch,
) -> None:
    repository, service, coordinator = ownership
    await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    monkeypatch.setattr(
        repository,
        "apply_artifact_owner_operation",
        lambda operation: (_ for _ in ()).throw(
            CitationPersistenceUnavailable("private token value")
        ),
    )

    result = coordinator.reconcile_pending(limit=1)

    assert result.reason_codes == ("artifact_reconciliation_failed",)
    assert _outbox(service)[0].error_code == "artifact_reconciliation_failed"


@pytest.mark.asyncio
async def test_disabled_policy_preserves_ordinary_save_without_outbox_or_lease(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    enabled_repository = _repository(db)
    _persist(db, enabled_repository)
    disabled_repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        identity_context=_identity(db),
        fingerprint_codec=None,
    )
    service = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "chatbooks.json",
    )
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=disabled_repository,
    )
    service.set_citation_ownership_coordinator(coordinator)

    created = await service.create_chatbook(name="Still saved")

    assert created["name"] == "Still saved"
    assert _outbox(service) == []
    assert coordinator.reconcile_pending(limit=1).disabled is True
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == 0
    )
