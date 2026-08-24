from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.source_operation_store import (
    ResearchSourceOperationStore,
    SourceOperationConflictError,
)
from tldw_chatbook.Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
    SourceOperationValidationError,
)
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)


NOW = "2026-08-24T12:00:00Z"


def _operation(**changes: object) -> ResearchSourceOperation:
    values: dict[str, object] = {
        "operation_id": "operation-1",
        "idempotency_key": "local:workspace-1:source-1",
        "data_source": "local",
        "workspace_id": "workspace-1",
        "ingest_job_id": "",
        "canonical_item_type": CanonicalItemType.LOCAL_LIBRARY,
        "desired_selected": True,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(changes)
    return ResearchSourceOperation(**values)


def _store(tmp_path: Path) -> tuple[WorkspaceDB, ResearchSourceOperationStore]:
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    return db, ResearchSourceOperationStore(db)


def test_create_get_and_restart_round_trip_frozen_operation(tmp_path: Path) -> None:
    db, store = _store(tmp_path)
    operation = _operation()

    created = store.create(operation)

    assert created == operation
    assert store.get(operation.operation_id) == operation
    with pytest.raises(AttributeError):
        created.workspace_id = "different"  # type: ignore[misc]
    db.close()

    reopened_db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    reopened_store = ResearchSourceOperationStore(reopened_db)
    assert reopened_store.get(operation.operation_id) == operation
    reopened_db.close()


def test_create_rejects_duplicate_idempotency_key_with_typed_conflict(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    store.create(_operation())

    with pytest.raises(SourceOperationConflictError, match="idempotency"):
        store.create(
            _operation(
                operation_id="operation-2",
                workspace_id="workspace-2",
            )
        )
    db.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": 99},
        {"ingest_job_id": "ingest-job-99"},
        {"canonical_item_id": "canonical-99"},
        {"workspace_source_id": "membership-99"},
        {"catalog_status": SourceOperationStatus.IN_PROGRESS},
        {
            "catalog_status": SourceOperationStatus.FAILED,
            "error_stage": SourceOperationStage.CATALOG,
            "error_code": "catalog_failed",
            "error_message": "Safe diagnostic.",
        },
        {
            "revision": 99,
            "ingest_job_id": "ingest-job-99",
            "canonical_item_id": "canonical-99",
            "workspace_source_id": "membership-99",
            "catalog_status": SourceOperationStatus.SUCCEEDED,
            "association_status": SourceOperationStatus.SUCCEEDED,
            "readiness_status": SourceOperationStatus.SUCCEEDED,
        },
    ],
)
def test_create_accepts_only_pristine_durable_intent(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    db, store = _store(tmp_path)
    operation = _operation(**changes)

    with pytest.raises(SourceOperationValidationError, match="pristine"):
        store.create(operation)

    assert store.get(operation.operation_id) is None
    db.close()


def test_catalog_start_records_the_ingest_job_link_after_operation_create(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    operation = store.create(_operation(ingest_job_id=""))

    started = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id="ingest-job-9",
    )

    assert started.ingest_job_id == "ingest-job-9"
    assert store.get(operation.operation_id) == started
    db.close()


def test_list_incomplete_is_bounded_and_excludes_fully_succeeded_rows(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    for index in range(3):
        store.create(
            _operation(
                operation_id=f"operation-{index}",
                idempotency_key=f"local:workspace-1:source-{index}",
            )
        )
    complete = store.get("operation-0")
    assert complete is not None
    complete = store.advance_stage(
        complete.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=complete.revision,
        canonical_item_id="101",
        timestamp="2026-08-24T12:01:00Z",
    )
    complete = store.advance_stage(
        complete.operation_id,
        stage=SourceOperationStage.ASSOCIATION,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=complete.revision,
        workspace_source_id="membership-101",
        timestamp="2026-08-24T12:02:00Z",
    )
    store.advance_stage(
        complete.operation_id,
        stage=SourceOperationStage.READINESS,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=complete.revision,
        timestamp="2026-08-24T12:03:00Z",
    )

    page = store.list_incomplete(limit=1, offset=1)

    assert [item.operation_id for item in page] == ["operation-2"]
    with pytest.raises(SourceOperationValidationError, match="limit"):
        store.list_incomplete(limit=0)
    with pytest.raises(SourceOperationValidationError, match="limit"):
        store.list_incomplete(limit=101)
    with pytest.raises(SourceOperationValidationError, match="offset"):
        store.list_incomplete(offset=-1)
    db.close()


def test_list_incomplete_does_not_starve_pending_behind_failed_rows(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    for index in range(100):
        operation = store.create(
            _operation(
                operation_id=f"operation-failed-{index:03d}",
                idempotency_key=f"local:workspace-1:failed-{index:03d}",
            )
        )
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.FAILED,
            expected_revision=operation.revision,
            error_code="catalog_failed",
            error_message="Safe retryable failure.",
        )
    pending = store.create(
        _operation(
            operation_id="operation-z-pending",
            idempotency_key="local:workspace-1:z-pending",
        )
    )

    page = store.list_incomplete(limit=100)

    assert pending.operation_id in {operation.operation_id for operation in page}
    db.close()


def test_list_association_actionable_applies_bound_after_stage_filtering(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    for index in range(55):
        readiness_only = store.create(
            _operation(
                operation_id=f"operation-readiness-{index:03d}",
                idempotency_key=f"local:workspace-1:readiness-{index:03d}",
                created_at=f"2026-08-24T10:{index // 60:02d}:{index % 60:02d}Z",
                updated_at=f"2026-08-24T10:{index // 60:02d}:{index % 60:02d}Z",
            )
        )
        readiness_only = store.advance_stage(
            readiness_only.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=readiness_only.revision,
            canonical_item_id=str(1000 + index),
        )
        store.advance_stage(
            readiness_only.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=readiness_only.revision,
            workspace_source_id=f"membership-{index}",
        )
    store.create(
        _operation(
            operation_id="operation-pristine-not-actionable",
            idempotency_key="local:workspace-1:pristine-not-actionable",
            created_at="2026-08-24T09:00:00Z",
            updated_at="2026-08-24T09:00:00Z",
        )
    )

    failed = store.create(
        _operation(
            operation_id="operation-explicit-failed",
            idempotency_key="local:workspace-1:explicit-failed",
            created_at="2026-08-24T11:58:00Z",
            updated_at="2026-08-24T11:58:00Z",
        )
    )
    store.advance_stage(
        failed.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.FAILED,
        expected_revision=failed.revision,
        error_code="catalog_failed",
        error_message="Safe retryable failure.",
    )
    later_b = store.create(
        _operation(
            operation_id="operation-actionable-b",
            idempotency_key="local:workspace-1:actionable-b",
            created_at="2026-08-24T12:00:00Z",
            updated_at="2026-08-24T12:00:00Z",
        )
    )
    later_a = store.create(
        _operation(
            operation_id="operation-actionable-a",
            idempotency_key="local:workspace-1:actionable-a",
            created_at="2026-08-24T12:00:00Z",
            updated_at="2026-08-24T12:00:00Z",
        )
    )
    later_b = store.advance_stage(
        later_b.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=later_b.revision,
        ingest_job_id="ingest-job-actionable-b",
    )
    later_a = store.advance_stage(
        later_a.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=later_a.revision,
        canonical_item_id="9001",
    )

    page = store.list_association_actionable(limit=2)

    assert [operation.operation_id for operation in page] == [
        later_a.operation_id,
        later_b.operation_id,
    ]
    with pytest.raises(SourceOperationValidationError, match="limit"):
        store.list_association_actionable(limit=0)
    with pytest.raises(SourceOperationValidationError, match="limit"):
        store.list_association_actionable(limit=101)
    db.close()


def test_advance_stage_enforces_order_revision_and_explicit_named_retry(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    operation = store.create(_operation())

    with pytest.raises(SourceOperationValidationError, match="catalog"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=operation.revision,
            workspace_source_id="membership-1",
        )

    catalog = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=operation.revision,
        canonical_item_id="101",
    )
    with pytest.raises(SourceOperationConflictError, match="revision"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.FAILED,
            expected_revision=operation.revision,
            error_code="association_unavailable",
            error_message="Workspace association is unavailable.",
        )

    failed = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.ASSOCIATION,
        status=SourceOperationStatus.FAILED,
        expected_revision=catalog.revision,
        error_code="association_unavailable",
        error_message="Workspace association is unavailable.",
    )
    assert failed.error_stage is SourceOperationStage.ASSOCIATION
    assert failed.catalog_status is SourceOperationStatus.SUCCEEDED

    with pytest.raises(SourceOperationValidationError, match="retry_failed_stage"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.PENDING,
            expected_revision=failed.revision,
        )
    with pytest.raises(SourceOperationConflictError, match="failed stage"):
        store.retry_failed_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            expected_revision=failed.revision,
        )

    retried = store.retry_failed_stage(
        operation.operation_id,
        stage=SourceOperationStage.ASSOCIATION,
        expected_revision=failed.revision,
    )
    assert retried.association_status is SourceOperationStatus.PENDING
    assert retried.error_stage is None
    assert retried.error_code == ""
    assert retried.error_message == ""
    assert retried.catalog_status is SourceOperationStatus.SUCCEEDED
    db.close()


def test_later_stages_cannot_retarget_canonical_or_association_identity(
    tmp_path: Path,
) -> None:
    db, store = _store(tmp_path)
    operation = store.create(_operation())
    catalog = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=operation.revision,
        canonical_item_id="101",
    )

    with pytest.raises(SourceOperationValidationError, match="canonical_item_id"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=catalog.revision,
            canonical_item_id="202",
            workspace_source_id="membership-101",
        )

    associated = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.ASSOCIATION,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=catalog.revision,
        workspace_source_id="membership-101",
    )
    with pytest.raises(SourceOperationValidationError, match="workspace_source_id"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.READINESS,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=associated.revision,
            workspace_source_id="membership-202",
        )
    db.close()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"operation_id": ""}, "operation_id"),
        ({"workspace_id": "/Users/alice/private/source.txt"}, "private path"),
        ({"idempotency_key": "token=top-secret"}, "secret"),
        (
            {
                "data_source": "server",
                "canonical_item_type": CanonicalItemType.SERVER_MEDIA,
                "server_profile_id": "https://user:password@example.test",
            },
            "credential",
        ),
        ({"operation_id": "x" * 257}, "maximum"),
    ],
)
def test_operation_rejects_unbounded_or_sensitive_identity_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SourceOperationValidationError, match=message):
        _operation(**changes)


@pytest.mark.parametrize(
    "error_message",
    [
        "api_key=super-secret",
        "Failed reading /Users/alice/Private/source.txt",
        "Fetch failed for https://alice:password@example.test/source",
    ],
)
def test_operation_rejects_secret_path_or_credential_bearing_diagnostics(
    error_message: str,
) -> None:
    with pytest.raises(SourceOperationValidationError):
        _operation(
            catalog_status=SourceOperationStatus.SUCCEEDED,
            canonical_item_id="101",
            association_status=SourceOperationStatus.FAILED,
            error_stage=SourceOperationStage.ASSOCIATION,
            error_code="association_failed",
            error_message=error_message,
        )


@pytest.mark.parametrize(
    "error_message",
    [
        "/root/private/source.txt",
        "Failed reading /etc/passwd while cataloging",
        "Failed reading ../../private/source.txt",
        "Fetch failed for postgres://alice:password@example.test/catalog",
        "OPENAI_API_KEY=top-secret",
        "client_secret=top-secret",
    ],
)
def test_operation_rejects_reviewed_receipt_privacy_counterexamples(
    error_message: str,
) -> None:
    with pytest.raises(SourceOperationValidationError):
        _operation(
            catalog_status=SourceOperationStatus.FAILED,
            error_stage=SourceOperationStage.CATALOG,
            error_code="catalog_failed",
            error_message=error_message,
        )


def test_local_and_server_owner_id_invariants_are_validated() -> None:
    with pytest.raises(SourceOperationValidationError, match="Local"):
        _operation(server_profile_id="server-profile")
    with pytest.raises(SourceOperationValidationError, match="required"):
        _operation(
            data_source="server",
            canonical_item_type=CanonicalItemType.SERVER_MEDIA,
        )
    with pytest.raises(SourceOperationValidationError, match="canonical_item_type"):
        _operation(canonical_item_type=CanonicalItemType.SERVER_MEDIA)

    server = _operation(
        data_source="server",
        server_profile_id="server-profile",
        principal_id="principal@example.test",
        workspace_id="workspace-900",
        canonical_item_type=CanonicalItemType.SERVER_MEDIA,
    )
    assert server.data_source.value == "server"


def test_server_identity_contract_preserves_adapter_identifier_shapes() -> None:
    operation = _operation(
        operation_id="018f47ab-74f2-7d45-a73b-97a8fcb1e580",
        idempotency_key="server:profile-1:principal@example.test:workspace_42.7:source+9",
        data_source="server",
        server_profile_id="profile-1",
        principal_id="credential-fingerprint:test:9d8cbe900ad878341fffc769",
        workspace_id="workspace_42.7",
        canonical_item_type=CanonicalItemType.SERVER_MEDIA,
    )

    assert operation.principal_id.startswith("credential-fingerprint:test:")


@pytest.mark.parametrize(
    ("jwt_token", "expected_principal_id"),
    [
        (
            "header."
            "eyJzdWIiOiJodHRwczovL2lzc3Vlci5leGFtcGxlL3VzZXJzL2FsaWNlJTJGd2VzdCJ9."
            "signature",
            "jwt-sub:https://issuer.example/users/alice%2Fwest",
        ),
        (
            "header."
            "eyJzdWIiOiJodHRwczovL2lzc3Vlci5leGFtcGxlL-eUqOaIty_DqWzDqHZlJTJGd2VzdCJ9."
            "signature",
            "jwt-sub:https://issuer.example/用户/élève%2Fwest",
        ),
    ],
)
def test_server_identity_contract_preserves_actual_uri_like_authority_ids(
    jwt_token: str, expected_principal_id: str
) -> None:
    principal_id = event_principal_id_from_active_context(
        SimpleNamespace(
            auth_token=jwt_token,
            credential_source="test",
        )
    )
    assert principal_id == expected_principal_id

    operation = _operation(
        data_source="server",
        server_profile_id="https://server.example/api/v1",
        principal_id=principal_id,
        workspace_id="urn:workspace:teams/acme%2Fresearch?region=west",
        canonical_item_type=CanonicalItemType.SERVER_MEDIA,
    )

    assert operation.server_profile_id == "https://server.example/api/v1"
    assert operation.principal_id == principal_id
    assert operation.workspace_id.endswith("region=west")


@pytest.mark.parametrize(
    ("principal_id", "message"),
    [
        ("jwt-sub:alice\x01west", "control or format"),
        ("jwt-sub:alice\u202ewest", "control or format"),
        ("é" * 513, "UTF-8 bytes"),
    ],
)
def test_authority_identity_rejects_control_format_and_utf8_overflow(
    principal_id: str, message: str
) -> None:
    with pytest.raises(SourceOperationValidationError, match=message):
        _operation(
            data_source="server",
            server_profile_id="https://server.example/api/v1",
            principal_id=principal_id,
            workspace_id="workspace-1",
            canonical_item_type=CanonicalItemType.SERVER_MEDIA,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data_source": "cloud"}, "data_source"),
        ({"canonical_item_type": "note"}, "canonical_item_type"),
        ({"catalog_status": "complete"}, "catalog_status"),
        ({"desired_selected": 1}, "desired_selected"),
        ({"revision": 0}, "revision"),
        ({"created_at": "2026-08-24T12:00:00"}, "timezone"),
        (
            {
                "catalog_status": SourceOperationStatus.SUCCEEDED,
                "canonical_item_id": "101",
                "association_status": SourceOperationStatus.FAILED,
                "error_stage": SourceOperationStage.ASSOCIATION,
                "error_code": "UPPER CASE",
                "error_message": "Safe diagnostic.",
            },
            "error_code",
        ),
        (
            {
                "catalog_status": SourceOperationStatus.SUCCEEDED,
                "canonical_item_id": "101",
                "association_status": SourceOperationStatus.FAILED,
                "error_stage": SourceOperationStage.ASSOCIATION,
                "error_code": "association_failed",
                "error_message": "x" * 513,
            },
            "maximum",
        ),
    ],
)
def test_operation_bounds_every_enum_flag_revision_timestamp_and_diagnostic(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SourceOperationValidationError, match=message):
        _operation(**changes)


def test_store_rejects_invalid_stage_status_and_compare_version(tmp_path: Path) -> None:
    db, store = _store(tmp_path)
    operation = store.create(_operation())

    with pytest.raises(SourceOperationValidationError, match="stage"):
        store.advance_stage(
            operation.operation_id,
            stage="upload",  # type: ignore[arg-type]
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=operation.revision,
        )
    with pytest.raises(SourceOperationValidationError, match="status"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status="complete",  # type: ignore[arg-type]
            expected_revision=operation.revision,
        )
    with pytest.raises(SourceOperationValidationError, match="expected_revision"):
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=True,
        )
    db.close()
