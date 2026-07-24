from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import sqlite3
import threading

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
    SharedDatabaseArtifactOwnershipStore,
)
from tldw_chatbook.Chat.citation_payload_lifecycle import CitationPayloadLifecycle
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.console_save_targets import (
    CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS,
    console_chatbook_artifact_payload,
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


def _create_shared_database_artifact_tables(
    db: CharactersRAGDB,
    *,
    trace_on_delete: str = "RESTRICT",
    artifact_on_delete: str = "CASCADE",
) -> None:
    assert trace_on_delete in {"RESTRICT", "NO ACTION", "CASCADE"}
    assert artifact_on_delete in {
        "RESTRICT",
        "NO ACTION",
        "CASCADE",
        "SET NULL",
    }
    with db.transaction() as cursor:
        cursor.execute(
            """
            CREATE TABLE shared_test_artifacts(
                artifact_id TEXT NOT NULL,
                artifact_revision INTEGER NOT NULL,
                PRIMARY KEY(artifact_id, artifact_revision)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE shared_test_artifact_owners(
                profile_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                artifact_revision INTEGER NOT NULL,
                PRIMARY KEY(
                    profile_id, trace_id, artifact_id, artifact_revision
                ),
                FOREIGN KEY(profile_id, trace_id)
                    REFERENCES rag_citation_traces(profile_id, trace_id)
                    ON DELETE {trace_on_delete},
                FOREIGN KEY(artifact_id, artifact_revision)
                    REFERENCES shared_test_artifacts(
                        artifact_id, artifact_revision
                    )
                    ON DELETE {artifact_on_delete}
            )
            """.format(
                trace_on_delete=trace_on_delete,
                artifact_on_delete=artifact_on_delete,
            )
        )


class _SharedDatabaseArtifactStore:
    artifact_backend_mode = ArtifactBackendMode.SHARED_DATABASE
    artifact_store_id = "shared-test-artifacts-v1"
    artifact_table = "shared_test_artifacts"
    artifact_owner_table = "shared_test_artifact_owners"

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        fail_after_owner: bool = False,
        omit_artifact: bool = False,
    ) -> None:
        self.artifact_database = db
        self.fail_after_owner = fail_after_owner
        self.omit_artifact = omit_artifact
        self.seen_connection: sqlite3.Connection | None = None

    def apply_shared_database_owner_mutation(
        self,
        cursor: sqlite3.Cursor,
        operation: ArtifactOwnerOperation,
    ) -> None:
        self.seen_connection = cursor.connection
        binding = operation.binding
        if operation.operation_kind is ArtifactOwnerOperationKind.LINK:
            if not self.omit_artifact:
                cursor.execute(
                    """
                    INSERT INTO shared_test_artifacts(
                        artifact_id, artifact_revision
                    ) VALUES (?, ?)
                    """,
                    (binding.artifact_id, binding.artifact_revision),
                )
            cursor.execute(
                """
                INSERT INTO shared_test_artifact_owners(
                    profile_id, trace_id, artifact_id, artifact_revision
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    binding.profile_id,
                    binding.trace_id,
                    binding.artifact_id,
                    binding.artifact_revision,
                ),
            )
        else:
            cursor.execute(
                """
                DELETE FROM shared_test_artifact_owners
                WHERE profile_id = ? AND trace_id = ?
                  AND artifact_id = ? AND artifact_revision = ?
                """,
                (
                    binding.profile_id,
                    binding.trace_id,
                    binding.artifact_id,
                    binding.artifact_revision,
                ),
            )
            cursor.execute(
                """
                DELETE FROM shared_test_artifacts
                WHERE artifact_id = ? AND artifact_revision = ?
                """,
                (binding.artifact_id, binding.artifact_revision),
            )
        if self.fail_after_owner:
            raise RuntimeError("shared mutation interrupted")


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


def test_shared_database_owner_commit_uses_repository_transaction_and_real_fks(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _create_shared_database_artifact_tables(db)
    store = _SharedDatabaseArtifactStore(db)
    assert isinstance(store, SharedDatabaseArtifactOwnershipStore)
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=store,
        trace_repository=repository,
    )
    operation = coordinator.prepare_link_operation(
        _owner_request(repository),
        artifact_id="shared-artifact-1",
        artifact_revision=1,
        artifact_body='{"artifact":"shared-artifact-1"}',
    )

    coordinator.apply_shared_database_owner_operation(operation)

    connection = db.get_connection()
    assert store.seen_connection is connection
    assert (
        connection.execute("SELECT count(*) FROM shared_test_artifacts").fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT count(*) FROM shared_test_artifact_owners"
        ).fetchone()[0]
        == 1
    )


def test_shared_database_owner_failure_rolls_back_artifact_and_fk_owner(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _create_shared_database_artifact_tables(db)
    store = _SharedDatabaseArtifactStore(db, fail_after_owner=True)
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=store,
        trace_repository=repository,
    )
    operation = coordinator.prepare_link_operation(
        _owner_request(repository),
        artifact_id="shared-artifact-1",
        artifact_revision=1,
        artifact_body='{"artifact":"shared-artifact-1"}',
    )

    with pytest.raises(RuntimeError, match="shared mutation interrupted"):
        coordinator.apply_shared_database_owner_operation(operation)

    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM shared_test_artifacts").fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT count(*) FROM shared_test_artifact_owners"
        ).fetchone()[0]
        == 0
    )


def test_shared_database_owner_fk_is_enforced_in_the_shared_transaction(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _create_shared_database_artifact_tables(db)
    store = _SharedDatabaseArtifactStore(db, omit_artifact=True)
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=store,
        trace_repository=repository,
    )
    operation = coordinator.prepare_link_operation(
        _owner_request(repository),
        artifact_id="missing-shared-artifact",
        artifact_revision=1,
        artifact_body='{"artifact":"missing-shared-artifact"}',
    )

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        coordinator.apply_shared_database_owner_operation(operation)

    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM shared_test_artifact_owners")
        .fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    ("trace_on_delete", "artifact_on_delete"),
    [
        ("CASCADE", "CASCADE"),
        ("RESTRICT", "SET NULL"),
    ],
)
def test_shared_database_contract_rejects_destructive_or_ambiguous_fk_actions(
    db: CharactersRAGDB,
    trace_on_delete: str,
    artifact_on_delete: str,
) -> None:
    repository = _repository(db)
    _create_shared_database_artifact_tables(
        db,
        trace_on_delete=trace_on_delete,
        artifact_on_delete=artifact_on_delete,
    )

    with pytest.raises(ValueError, match="shared_database_owner_contract_required"):
        CitationArtifactOwnershipCoordinator(
            artifact_store=_SharedDatabaseArtifactStore(db),
            trace_repository=repository,
        )


@pytest.mark.parametrize("shadow_schema", ["temp", "attached"])
def test_shared_database_contract_rejects_non_main_table_shadows(
    db: CharactersRAGDB,
    tmp_path,
    shadow_schema: str,
) -> None:
    repository = _repository(db)
    _create_shared_database_artifact_tables(db)
    connection = db.get_connection()
    if shadow_schema == "attached":
        connection.execute(
            "ATTACH DATABASE ? AS shadow", (str(tmp_path / "shadow.db"),)
        )
        schema = "shadow"
    else:
        schema = "temp"
    connection.execute(
        f"""
        CREATE TABLE {schema}.shared_test_artifact_owners(
            artifact_id TEXT,
            artifact_revision INTEGER
        )
        """
    )

    with pytest.raises(ValueError, match="shared_database_owner_contract_required"):
        CitationArtifactOwnershipCoordinator(
            artifact_store=_SharedDatabaseArtifactStore(db),
            trace_repository=repository,
        )


def test_shared_database_real_owner_fk_prevents_live_trace_collection(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _create_shared_database_artifact_tables(db)
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=_SharedDatabaseArtifactStore(db),
        trace_repository=repository,
    )
    operation = coordinator.prepare_link_operation(
        _owner_request(repository),
        artifact_id="shared-artifact-live",
        artifact_revision=1,
        artifact_body='{"artifact":"shared-artifact-live"}',
    )
    coordinator.apply_shared_database_owner_operation(operation)
    _mark_owner_deleted(db)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        CitationPayloadLifecycle(repository, retention_policy=_policy()).collect(
            now=NOW
        )

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE trace_id = ?",
            (operation.binding.trace_id,),
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT count(*) FROM shared_test_artifact_owners"
        ).fetchone()[0]
        == 1
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
            artifact_body_fingerprint="hmac-sha256-v1:body",
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
            artifact_body_fingerprint="hmac-sha256-v1:body",
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
async def test_artifact_binding_macs_canonical_persisted_console_representation(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    payload = console_chatbook_artifact_payload(
        title="Unicode \u00e9vidence",
        message_text=("界" * (CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS + 7)),
        message_role="Assistant",
    )
    await service.create_chatbook(
        **payload,
        provenance_owner_request=_owner_request(repository),
    )
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    binding = ArtifactOwnerBinding.model_validate(
        registry["records"][0]["provenance_owner"],
        strict=True,
    )

    assert binding.artifact_body_fingerprint.startswith("hmac-sha256-v1:")
    assert "界" not in binding.artifact_body_fingerprint
    assert (
        len(registry["records"][0]["metadata"]["content"])
        == CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS
    )

    # Whitespace, key ordering, and escaped-vs-literal Unicode do not alter the
    # exact JSON value represented by the registry.
    service.registry_path.write_text(
        json.dumps(registry, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    result = coordinator.reconcile_pending(limit=1)
    assert result.completed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tampered_field", ["name", "metadata"])
async def test_same_revision_artifact_body_tamper_cannot_create_a_live_lease(
    ownership,
    tampered_field: str,
) -> None:
    repository, service, coordinator = ownership
    await service.create_chatbook(
        name="Grounded answer",
        metadata={"content": "Answer [S1].", "nested": {"a": 1, "b": 2}},
        provenance_owner_request=_owner_request(repository),
    )
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    if tampered_field == "name":
        registry["records"][0]["name"] = "Changed without a revision"
    else:
        registry["records"][0]["metadata"]["nested"]["a"] = 99
    service.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = coordinator.reconcile_pending(limit=1)

    assert result.completed == 0
    assert result.failed == 1
    assert (
        repository.db.get_connection()
        .execute("SELECT count(*) FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == 0
    )


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
async def test_grounded_artifact_update_without_request_unlinks_old_revision(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )

    replaced = await service.update_chatbook(
        created["chatbook_id"],
        name="Edited without grounded provenance",
    )

    operations = _outbox(service)
    assert replaced["artifact_revision"] == 2
    assert [
        (item.binding.artifact_revision, item.operation_kind) for item in operations
    ] == [
        (1, ArtifactOwnerOperationKind.LINK),
        (1, ArtifactOwnerOperationKind.UNLINK),
    ]
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    assert "provenance_owner" not in registry["records"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable_mode", ["disabled", "key_unavailable"])
@pytest.mark.parametrize("mutation", ["update", "delete"])
async def test_unavailable_owner_writes_defer_unlink_until_recovery(
    db: CharactersRAGDB,
    tmp_path,
    unavailable_mode: str,
    mutation: str,
) -> None:
    enabled_repository = _repository(db)
    _persist(db, enabled_repository)
    service = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "chatbooks.json",
    )
    enabled = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=enabled_repository,
    )
    service.set_citation_ownership_coordinator(enabled)
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(enabled_repository),
    )
    assert enabled.reconcile_pending(limit=1).completed == 1

    unavailable_repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(
            canonical_writes_enabled=unavailable_mode != "disabled"
        ),
        identity_context=_identity(db),
        fingerprint_codec=None,
    )
    unavailable = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=unavailable_repository,
    )
    service.set_citation_ownership_coordinator(unavailable)

    if mutation == "update":
        changed = await service.update_chatbook(
            created["chatbook_id"],
            name="Ordinary ungrounded edit",
        )
        assert changed["artifact_revision"] == 2
    else:
        assert await service.delete_chatbook(created["chatbook_id"]) is True

    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    assert len(registry["provenance_outbox"]) == 1
    assert registry["provenance_outbox"][0]["entry_kind"] == "deferred_unlink"
    assert (
        db.get_connection()
        .execute("SELECT state FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == "live"
    )
    if unavailable_mode == "disabled":
        assert unavailable.reconcile_pending(limit=1).disabled is True
    else:
        assert unavailable.reconcile_pending(limit=1).completed == 0

    recovered = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=enabled_repository,
    )
    service.set_citation_ownership_coordinator(recovered)
    assert recovered.reconcile_pending(limit=1).completed == 1
    assert _outbox(service) == []
    assert (
        db.get_connection()
        .execute("SELECT state FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == "released"
    )


@pytest.mark.asyncio
async def test_invalid_deferred_unlink_is_quarantined_without_starving_next(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    enabled_repository = _repository(db)
    _persist(db, enabled_repository)
    service = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "chatbooks.json",
    )
    enabled = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=enabled_repository,
    )
    service.set_citation_ownership_coordinator(enabled)
    artifacts = []
    for name in ("first", "second"):
        artifacts.append(
            await service.create_chatbook(
                name=name,
                provenance_owner_request=_owner_request(enabled_repository),
            )
        )
        assert enabled.reconcile_pending(limit=1).completed == 1

    disabled_repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        identity_context=_identity(db),
        fingerprint_codec=None,
    )
    disabled = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=disabled_repository,
    )
    service.set_citation_ownership_coordinator(disabled)
    for artifact in artifacts:
        await service.delete_chatbook(artifact["chatbook_id"])

    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    binding_id = registry["provenance_outbox"][0]["binding"]["binding_id"]
    registry["provenance_outbox"][0]["binding"]["binding_id"] = binding_id[:-1] + (
        "0" if binding_id[-1] != "0" else "1"
    )
    service.registry_path.write_text(json.dumps(registry), encoding="utf-8")
    recovered = CitationArtifactOwnershipCoordinator(
        artifact_store=service,
        trace_repository=enabled_repository,
    )
    service.set_citation_ownership_coordinator(recovered)

    assert recovered.reconcile_pending(limit=1).completed == 0
    quarantined = json.loads(service.registry_path.read_text(encoding="utf-8"))
    assert (
        quarantined["provenance_outbox"][0]["error_code"]
        == "artifact_owner_binding_invalid"
    )
    assert recovered.reconcile_pending(limit=1).completed == 1
    lease_states = {
        row["artifact_id"]: row["state"]
        for row in db.get_connection()
        .execute("SELECT artifact_id, state FROM rag_artifact_owner_leases")
        .fetchall()
    }
    assert lease_states == {
        str(artifacts[0]["chatbook_id"]): "live",
        str(artifacts[1]["chatbook_id"]): "released",
    }


@pytest.mark.asyncio
async def test_verified_binding_tamper_fails_closed_without_deleting_artifact(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    assert coordinator.reconcile_pending(limit=1).completed == 1
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    binding_id = registry["records"][0]["provenance_owner"]["binding_id"]
    registry["records"][0]["provenance_owner"]["binding_id"] = binding_id[:-1] + (
        "0" if binding_id[-1] != "0" else "1"
    )
    service.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="artifact_owner_binding_invalid",
    ):
        await service.delete_chatbook(created["chatbook_id"])

    assert (await service.get_chatbook(created["chatbook_id"]))["name"] == (
        "Grounded answer"
    )


@pytest.mark.asyncio
async def test_stale_replacement_request_still_unlinks_old_revision(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    created = await service.create_chatbook(
        name="Grounded answer",
        provenance_owner_request=_owner_request(repository),
    )
    stale_request = _owner_request(repository)
    _mark_owner_deleted(repository.db)

    replaced = await service.update_chatbook(
        created["chatbook_id"],
        name="Edited after provenance became stale",
        provenance_owner_request=stale_request,
    )

    assert replaced["artifact_revision"] == 2
    assert [item.operation_kind for item in _outbox(service)] == [
        ArtifactOwnerOperationKind.LINK,
        ArtifactOwnerOperationKind.UNLINK,
    ]
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    assert "provenance_owner" not in registry["records"][0]


def test_concurrent_grounded_replacements_serialize_revision_lease_pairs(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    created = asyncio.run(
        service.create_chatbook(
            name="Grounded answer",
            provenance_owner_request=_owner_request(repository),
        )
    )
    requests = [_owner_request(repository), _owner_request(repository)]

    def replace(index: int) -> dict:
        return asyncio.run(
            service.update_chatbook(
                created["chatbook_id"],
                name=f"Grounded replacement {index}",
                provenance_owner_request=requests[index],
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        replacements = list(executor.map(replace, range(2)))

    assert {item["artifact_revision"] for item in replacements} == {2, 3}
    assert [
        (item.binding.artifact_revision, item.operation_kind)
        for item in _outbox(service)
    ] == [
        (1, ArtifactOwnerOperationKind.LINK),
        (1, ArtifactOwnerOperationKind.UNLINK),
        (2, ArtifactOwnerOperationKind.LINK),
        (2, ArtifactOwnerOperationKind.UNLINK),
        (3, ArtifactOwnerOperationKind.LINK),
    ]


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


def test_collection_barrier_registration_is_idempotent_for_same_registry(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    repository = _repository(db)
    first = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "chatbooks.json",
    )
    CitationArtifactOwnershipCoordinator(
        artifact_store=first,
        trace_repository=repository,
    )
    rewired = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "./chatbooks.json",
    )

    CitationArtifactOwnershipCoordinator(
        artifact_store=rewired,
        trace_repository=repository,
    )


def test_different_registry_cannot_replace_an_existing_collection_barrier(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    first = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "first.json",
    )
    first_coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=first,
        trace_repository=repository,
    )
    first.set_citation_ownership_coordinator(first_coordinator)
    asyncio.run(
        first.create_chatbook(
            name="Pending owner in registry A",
            provenance_owner_request=_owner_request(repository),
        )
    )
    second = LocalChatbookService(
        db_paths={},
        registry_path=tmp_path / "second.json",
    )

    with pytest.raises(
        ValueError, match="artifact_collection_barrier_already_registered"
    ):
        CitationArtifactOwnershipCoordinator(
            artifact_store=second,
            trace_repository=repository,
        )

    _mark_owner_deleted(db)
    result = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).collect(now=NOW)
    assert result.traces_collected == 0
    assert len(_outbox(first)) == 1


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
async def test_limit_one_reconciliation_cursor_survives_restart_and_skips_failure(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    await service.create_chatbook(
        name="Permanently tampered first artifact",
        provenance_owner_request=_owner_request(repository),
    )
    second = await service.create_chatbook(
        name="Valid second artifact",
        provenance_owner_request=_owner_request(repository),
    )
    registry = json.loads(service.registry_path.read_text(encoding="utf-8"))
    registry["records"][0]["name"] = "Tampered without revision"
    service.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    first_result = coordinator.reconcile_pending(limit=1)
    assert (first_result.completed, first_result.failed, first_result.examined) == (
        0,
        1,
        1,
    )

    restarted_service = LocalChatbookService(
        db_paths={},
        registry_path=service.registry_path,
    )
    restarted = CitationArtifactOwnershipCoordinator(
        artifact_store=restarted_service,
        trace_repository=repository,
    )
    restarted_service.set_citation_ownership_coordinator(restarted)
    second_result = restarted.reconcile_pending(limit=1)

    assert (second_result.completed, second_result.failed) == (1, 0)
    live_artifacts = {
        row["artifact_id"]
        for row in repository.db.get_connection()
        .execute(
            "SELECT artifact_id FROM rag_artifact_owner_leases WHERE state = 'live'"
        )
        .fetchall()
    }
    assert live_artifacts == {str(second["chatbook_id"])}


@pytest.mark.asyncio
async def test_fair_cursor_never_selects_unlink_before_its_pending_link(
    ownership,
) -> None:
    repository, service, coordinator = ownership
    created = await service.create_chatbook(
        name="Deleted before first reconciliation",
        provenance_owner_request=_owner_request(repository),
    )
    await service.delete_chatbook(created["chatbook_id"])

    assert coordinator.reconcile_pending(limit=1).completed == 1
    assert (
        repository.db.get_connection()
        .execute("SELECT state FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == "live"
    )
    assert coordinator.reconcile_pending(limit=1).completed == 1
    assert (
        repository.db.get_connection()
        .execute("SELECT state FROM rag_artifact_owner_leases")
        .fetchone()[0]
        == "released"
    )


def test_artifact_phase_one_precedes_collection_and_holds_pending_barrier(
    ownership,
    monkeypatch,
) -> None:
    repository, service, _coordinator = ownership
    request = _owner_request(repository)
    registry_written = threading.Event()
    release_artifact = threading.Event()
    collection_started = threading.Event()
    collection_done = threading.Event()
    original_save = service._save_registry
    result: dict[str, object] = {}

    def pause_after_registry_write(payload: dict) -> None:
        original_save(payload)
        registry_written.set()
        assert release_artifact.wait(5)

    def create_artifact() -> None:
        result["artifact"] = asyncio.run(
            service.create_chatbook(
                name="Grounded answer",
                provenance_owner_request=request,
            )
        )

    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())

    def collect() -> None:
        collection_started.set()
        result["collection"] = lifecycle.collect(now=NOW)
        collection_done.set()

    monkeypatch.setattr(service, "_save_registry", pause_after_registry_write)
    artifact_thread = threading.Thread(target=create_artifact)
    artifact_thread.start()
    assert registry_written.wait(5)
    _mark_owner_deleted(repository.db)
    collection_thread = threading.Thread(target=collect)
    collection_thread.start()
    assert collection_started.wait(5)
    assert not collection_done.wait(0.1)

    release_artifact.set()
    artifact_thread.join(5)
    collection_thread.join(5)

    assert not artifact_thread.is_alive()
    assert not collection_thread.is_alive()
    assert result["collection"].traces_collected == 0
    assert len(_outbox(service)) == 1


def test_collection_precedes_artifact_phase_one_and_stale_request_saves_ungrounded(
    ownership,
    monkeypatch,
) -> None:
    repository, service, _coordinator = ownership
    request = _owner_request(repository)
    _mark_owner_deleted(repository.db)
    collection_in_transaction = threading.Event()
    release_collection = threading.Event()
    artifact_started = threading.Event()
    artifact_done = threading.Event()
    result: dict[str, object] = {}
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())

    def pause_before_delete(*_args) -> None:
        collection_in_transaction.set()
        assert release_collection.wait(5)

    def collect() -> None:
        result["collection"] = lifecycle.collect(now=NOW)

    def create_artifact() -> None:
        artifact_started.set()
        try:
            result["artifact"] = asyncio.run(
                service.create_chatbook(
                    name="Ordinary saved answer",
                    provenance_owner_request=request,
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            result["artifact_error"] = exc
        finally:
            artifact_done.set()

    monkeypatch.setattr(lifecycle, "_before_collect_delete", pause_before_delete)
    collection_thread = threading.Thread(target=collect)
    collection_thread.start()
    assert collection_in_transaction.wait(5)
    artifact_thread = threading.Thread(target=create_artifact)
    artifact_thread.start()
    assert artifact_started.wait(5)
    assert not artifact_done.wait(0.1)

    release_collection.set()
    collection_thread.join(5)
    artifact_thread.join(5)

    assert not collection_thread.is_alive()
    assert not artifact_thread.is_alive()
    assert result["collection"].traces_collected == 1
    assert "artifact_error" not in result
    assert result["artifact"]["name"] == "Ordinary saved answer"
    assert _outbox(service) == []


@pytest.mark.asyncio
async def test_stale_owner_request_does_not_block_ordinary_artifact_save(
    ownership,
) -> None:
    repository, service, _coordinator = ownership
    request = _owner_request(repository)
    _mark_owner_deleted(repository.db)

    created = await service.create_chatbook(
        name="Ordinary saved answer",
        provenance_owner_request=request,
    )

    assert created["name"] == "Ordinary saved answer"
    assert _outbox(service) == []


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
