from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta

import pytest
from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileRecord,
    ProfileScope,
    RecordState,
    ScopeKind,
    SyncMode,
    WorkingContextPayload,
)

from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.key_protector import ProfileLockedError
from tldw_chatbook.Personal_Context.runtime_policy import (
    AgentAuthority,
    PersonalContextAuthorityError,
)
from tldw_chatbook.Personal_Context.service import (
    PersonalContextSettingsSnapshot,
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
    RecordMutation,
    SettingsScopeSnapshot,
)


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


@pytest.fixture
def service(tmp_path, memory_protector):
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    return PersonalContextService(repository, clock=lambda: NOW, id_factory=Ids())


def _preference(
    service: PersonalContextService,
    *,
    value: str = "concise",
    subject: str = "response.detail",
):
    manifest = service.get_manifest()
    scope = service.list_scopes()[0]
    return ProfileRecord(
        profile_id=manifest.profile_id,
        record_id=f"record-{value}",
        scope_id=scope.scope_id,
        kind="preference",
        payload=PreferencePayload(subject=subject, polarity="like", value=value),
        semantic_key={"namespace": "preference", "subject": subject},
        state="active",
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        provenance={
            "source": "manual",
            "actor": "user",
            "reason_code": "settings_edit",
        },
        version_id=f"version-{value}",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )


def test_create_profile_atomically_persists_exactly_one_global_scope(service) -> None:
    manifest = service.create_profile()

    scopes = service.list_scopes()
    assert service.get_manifest() == manifest
    assert len(scopes) == 1
    assert scopes[0].kind is ScopeKind.GLOBAL
    assert scopes[0].profile_id == manifest.profile_id


def test_update_rejects_record_kind_changes(service) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=WorkingContextPayload(subject="current task", value="draft"),
        semantic_key={"namespace": "working_context", "subject": "current task"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        no_expiry=True,
    )

    with pytest.raises(ValueError, match="kind cannot be changed"):
        service.update_record(
            current.record_id,
            RecordMutation(
                payload=PreferencePayload(
                    subject="response.detail", polarity="like", value="concise"
                ),
                semantic_key={
                    "namespace": "preference",
                    "subject": "response.detail",
                },
            ),
            expected_version_id=current.version_id,
        )


def test_working_context_defaults_to_bounded_retention_and_can_be_no_expiry(
    service,
) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]

    bounded = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=WorkingContextPayload(subject="current task", value="draft"),
        semantic_key={"namespace": "working_context", "subject": "current task"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    forever = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=WorkingContextPayload(subject="long term", value="keep"),
        semantic_key={"namespace": "working_context", "subject": "long term"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        no_expiry=True,
    )

    assert bounded.expires_at == NOW + timedelta(days=30)
    assert bounded.no_expiry is False
    assert forever.expires_at is None
    assert forever.no_expiry is True


def test_working_context_edit_preserves_existing_expiry_when_unspecified(
    service,
) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]
    expires_at = NOW + timedelta(days=7)
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=WorkingContextPayload(subject="current task", value="draft"),
        semantic_key={"namespace": "working_context", "subject": "current task"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        expires_at=expires_at,
    )

    updated = service.update_record(
        current.record_id,
        RecordMutation(
            payload=WorkingContextPayload(subject="current task", value="revised")
        ),
        expected_version_id=current.version_id,
    )

    assert updated.expires_at == expires_at
    assert updated.no_expiry is False


def test_workspace_identity_and_label_exist_only_in_encrypted_binding(
    service,
) -> None:
    service.create_profile()
    scope = service.create_workspace_scope("workspace-private-id", "Secret Label")

    assert scope.kind is ScopeKind.WORKSPACE
    assert service.map_workspace_scope("workspace-private-id", scope.scope_id) == scope
    durable = service._repository.db_path.read_bytes()
    assert b"workspace-private-id" not in durable
    assert b"Secret Label" not in durable


def test_service_workspace_binding_api_accepts_only_exact_v1_bodies(
    service,
) -> None:
    manifest = service.create_profile()
    scope = ProfileScope(
        scope_id="legacy-workspace-scope",
        profile_id=manifest.profile_id,
        kind=ScopeKind.WORKSPACE,
        version_id="legacy-workspace-version",
        created_at=NOW,
        updated_at=NOW,
    )
    service._repository.commit_scope(scope)
    service._repository.commit_scope_binding(
        scope.scope_id,
        {
            "version": True,
            "local_workspace_id": "legacy-arbitrary-body",
            "label": "legacy-label",
        },
    )
    record = ProfileRecord.model_validate(
        {
            **_preference(service, value="legacy-binding").model_dump(mode="python"),
            "scope_id": scope.scope_id,
        }
    )

    assert service.get_workspace_binding(scope.scope_id) is None
    assert service.list_workspace_bindings() == {}
    with pytest.raises(ValueError, match="mapped"):
        service.create_record(record)
    service.set_runtime_enabled(True)
    with pytest.raises(PersonalContextAuthorityError, match="scope_unmapped"):
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    with service._repository._connect() as connection:
        reasons = {
            row[0]
            for row in connection.execute(
                "SELECT reason_code FROM quarantine "
                "WHERE object_type = 'scope_binding' AND object_id = ?",
                (scope.scope_id,),
            )
        }
    assert reasons == {"invalid_workspace_binding"}

    service.map_workspace_scope("legacy-workspace-local", scope.scope_id)
    assert service.get_workspace_binding(scope.scope_id) == {
        "version": 1,
        "local_workspace_id": "legacy-workspace-local",
        "label": "",
    }
    assert service.list_workspace_bindings() == {
        scope.scope_id: service.get_workspace_binding(scope.scope_id)
    }
    assert service.create_record(record) == record


def test_missing_encrypted_workspace_binding_fails_closed_as_unmapped(service) -> None:
    service.create_profile()
    scope = service.create_workspace_scope("missing-body-local", "Private Label")
    version = service._repository.get_scope_binding_version(scope.scope_id)
    assert version is not None
    with service._repository._connect() as connection:
        connection.execute(
            "DELETE FROM encrypted_objects WHERE object_type = 'scope_binding' "
            "AND object_id = ? AND version_id = ?",
            (scope.scope_id, version),
        )

    assert service.get_workspace_binding(scope.scope_id) is None
    with pytest.raises(ValueError, match="mapped"):
        service.create_record(
            ProfileRecord.model_validate(
                {
                    **_preference(service, value="missing-body").model_dump(
                        mode="python"
                    ),
                    "scope_id": scope.scope_id,
                }
            )
        )


def test_concurrent_workspace_creation_cannot_duplicate_local_binding(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    first.create_profile()
    second = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    barrier = threading.Barrier(2)
    for candidate in (first, second):
        original = candidate._repository.commit_scope_with_binding

        def commit_after_barrier(*args, _original=original, **kwargs):
            barrier.wait()
            return _original(*args, **kwargs)

        candidate._repository.commit_scope_with_binding = commit_after_barrier

    def create(candidate):
        try:
            return candidate.create_workspace_scope("same-local-id", "Private Label")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [
            future.result(timeout=5)
            for future in (
                executor.submit(create, first),
                executor.submit(create, second),
            )
        ]

    assert sum(isinstance(item, ProfileScope) for item in outcomes) == 1
    assert sum(isinstance(item, ValueError) for item in outcomes) == 1
    assert len(first.list_scopes()) == 2


def test_concurrent_workspace_mapping_cannot_duplicate_local_binding(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    manifest = first.create_profile()
    second = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    scopes = tuple(
        ProfileScope(
            scope_id=f"incoming-{index}",
            profile_id=manifest.profile_id,
            kind=ScopeKind.WORKSPACE,
            version_id=f"incoming-{index}-version",
            created_at=NOW,
            updated_at=NOW,
        )
        for index in range(2)
    )
    first._repository.commit_scope(scopes[0])
    first._repository.commit_scope(scopes[1])
    barrier = threading.Barrier(2)
    for candidate in (first, second):
        original = candidate._repository.commit_scope_binding

        def commit_after_barrier(*args, _original=original, **kwargs):
            barrier.wait()
            return _original(*args, **kwargs)

        candidate._repository.commit_scope_binding = commit_after_barrier

    def bind(candidate, scope):
        try:
            return candidate.map_workspace_scope("same-local-id", scope.scope_id)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [
            future.result(timeout=5)
            for future in (
                executor.submit(bind, first, scopes[0]),
                executor.submit(bind, second, scopes[1]),
            )
        ]

    assert sum(isinstance(item, ProfileScope) for item in outcomes) == 1
    assert sum(isinstance(item, ValueError) for item in outcomes) == 1


def test_record_lifecycle_increments_manifest_and_enforces_cas(service) -> None:
    service.create_profile()
    created = service.create_record(_preference(service))
    assert service.get_manifest().revision == 1

    changed = service.update_record(
        created.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value="detailed"
            )
        ),
        expected_version_id=created.version_id,
    )
    assert changed.parent_version_id == created.version_id
    assert changed.payload.value == "detailed"
    assert service.get_manifest().revision == 2

    with pytest.raises(ProfileConflictError):
        service.update_record(
            created.record_id,
            RecordMutation(payload=changed.payload),
            expected_version_id=created.version_id,
        )
    assert service.get_manifest().revision == 2


def test_same_scope_kind_and_active_key_collides_but_archived_does_not(service) -> None:
    service.create_profile()
    first = service.create_record(_preference(service, value="first"))
    with pytest.raises(ProfileKeyCollisionError) as caught:
        service.create_record(_preference(service, value="second"))
    assert caught.value.record_id == first.record_id

    service.archive_record(first.record_id, expected_version_id=first.version_id)
    second = service.create_record(_preference(service, value="second"))
    assert second.record_id != first.record_id


def test_concurrent_same_key_create_uses_pre_scan_manifest_fence(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first_service = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    first_service.create_profile()
    second_service = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    scan_barrier = threading.Barrier(2)
    winner_done = threading.Event()
    first_scan = first_service._require_no_collision
    second_scan = second_service._require_no_collision

    def scan_then_win(*args, **kwargs):
        first_scan(*args, **kwargs)
        scan_barrier.wait()

    def scan_then_wait(*args, **kwargs):
        second_scan(*args, **kwargs)
        scan_barrier.wait()
        assert winner_done.wait(5)

    first_service._require_no_collision = scan_then_win
    second_service._require_no_collision = scan_then_wait
    first_record = _preference(first_service, value="race-first")
    second_record = _preference(second_service, value="race-second")

    def create_winner():
        try:
            return first_service.create_record(first_record)
        finally:
            winner_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(create_winner)
        loser = executor.submit(second_service.create_record, second_record)
        assert winner.result(timeout=5) == first_record
        with pytest.raises(ProfileConflictError):
            loser.result(timeout=5)

    assert first_service.list_records(scope_ids=(first_record.scope_id,)) == (
        first_record,
    )


def test_concurrent_updates_cannot_create_duplicate_semantic_keys(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first_service = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    first_service.create_profile()
    first_record = first_service.create_record(
        _preference(first_service, value="first", subject="first-key")
    )
    second_record = first_service.create_record(
        _preference(first_service, value="second", subject="second-key")
    )
    second_service = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    scan_barrier = threading.Barrier(2)
    winner_done = threading.Event()
    first_scan = first_service._require_no_collision
    second_scan = second_service._require_no_collision

    def scan_then_win(*args, **kwargs):
        first_scan(*args, **kwargs)
        scan_barrier.wait()

    def scan_then_wait(*args, **kwargs):
        second_scan(*args, **kwargs)
        scan_barrier.wait()
        assert winner_done.wait(5)

    first_service._require_no_collision = scan_then_win
    second_service._require_no_collision = scan_then_wait
    mutation = RecordMutation(
        payload=PreferencePayload(subject="shared-key", polarity="like", value="same"),
        semantic_key={"namespace": "preference", "subject": "shared-key"},
    )

    def update_winner():
        try:
            return first_service.update_record(
                first_record.record_id,
                mutation,
                expected_version_id=first_record.version_id,
            )
        finally:
            winner_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(update_winner)
        loser = executor.submit(
            second_service.update_record,
            second_record.record_id,
            mutation,
            expected_version_id=second_record.version_id,
        )
        updated = winner.result(timeout=5)
        with pytest.raises(ProfileConflictError):
            loser.result(timeout=5)

    records = first_service.list_records(scope_ids=(first_record.scope_id,))
    assert updated.semantic_key is not None
    assert sum(record.semantic_key == updated.semantic_key for record in records) == 1


def test_archive_restore_delete_and_undo_are_immutable(service) -> None:
    service.create_profile()
    first = service.create_record(_preference(service))
    archived = service.archive_record(
        first.record_id, expected_version_id=first.version_id
    )
    assert archived.state is RecordState.ARCHIVED
    assert service.list_records(scope_ids=(first.scope_id,)) == ()
    assert service.list_records(scope_ids=(first.scope_id,), include_archived=True) == (
        archived,
    )

    restored = service.restore_record(
        first.record_id, expected_version_id=archived.version_id
    )
    deleted = service.delete_record(
        first.record_id, expected_version_id=restored.version_id
    )
    assert deleted.state is RecordState.DELETED
    assert deleted.payload is deleted.semantic_key is deleted.expires_at is None
    assert deleted.no_expiry is False

    undo_id = service.list_undo_ids()[0]
    undone = service.undo(undo_id)
    assert undone.state is RecordState.ACTIVE
    assert undone.parent_version_id == deleted.version_id
    assert service.list_undo_ids() == ()


def test_delete_retires_prior_record_and_outbox_content_but_keeps_bounded_undo(
    service,
) -> None:
    service.create_profile()
    first = ProfileRecord.model_validate(
        {
            **_preference(service, value="DELETE-FIRST-CANARY").model_dump(
                mode="python"
            ),
            "record_id": "record-delete",
            "version_id": "record-delete-v1",
        }
    )
    first = service.create_record(first)
    changed = service.update_record(
        first.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail",
                polarity="like",
                value="DELETE-LATEST-CANARY",
            )
        ),
        expected_version_id=first.version_id,
    )
    with service._repository._connect() as connection:
        prior_outbox = connection.execute(
            "SELECT outbox_id, envelope_version FROM encrypted_outbox "
            "WHERE object_type = 'record' AND object_id = ?",
            (first.record_id,),
        ).fetchall()
    assert len(prior_outbox) == 2

    tombstone = service.delete_record(
        first.record_id, expected_version_id=changed.version_id
    )

    with service._repository._connect() as connection:
        record_versions = connection.execute(
            "SELECT version_id FROM encrypted_objects "
            "WHERE object_type = 'record' AND object_id = ?",
            (first.record_id,),
        ).fetchall()
        remaining_outbox = connection.execute(
            "SELECT outbox_id, version_id, envelope_version FROM encrypted_outbox "
            "WHERE object_type = 'record' AND object_id = ?",
            (first.record_id,),
        ).fetchall()
        retired_outbox_envelopes = connection.execute(
            "SELECT COUNT(*) FROM encrypted_objects WHERE object_type = 'outbox' "
            f"AND object_id IN ({','.join('?' for _ in prior_outbox)})",
            tuple(row["outbox_id"] for row in prior_outbox),
        ).fetchone()[0]
        retired_outbox_heads = connection.execute(
            "SELECT COUNT(*) FROM object_heads WHERE object_type = 'outbox' "
            f"AND object_id IN ({','.join('?' for _ in prior_outbox)})",
            tuple(row["outbox_id"] for row in prior_outbox),
        ).fetchone()[0]
    assert [row["version_id"] for row in record_versions] == [tombstone.version_id]
    assert len(remaining_outbox) == 1
    assert remaining_outbox[0]["version_id"] == tombstone.version_id
    assert retired_outbox_envelopes == retired_outbox_heads == 0
    body = service._repository.get_outbox_body(remaining_outbox[0]["outbox_id"])
    assert body is not None
    assert body["record"]["state"] == "deleted"
    assert body["record"]["payload"] is None
    durable = service._repository.db_path.read_bytes()
    assert b"DELETE-FIRST-CANARY" not in durable
    assert b"DELETE-LATEST-CANARY" not in durable

    undo_id = service.list_undo_ids()[0]
    restored = service.undo(undo_id)
    assert restored.payload == changed.payload
    deleted_again = service.delete_record(
        restored.record_id, expected_version_id=restored.version_id
    )
    assert deleted_again.state is RecordState.DELETED
    expiring_undo = service.list_undo_ids()[0]
    service.clock = lambda: NOW + timedelta(hours=24)
    assert service.list_undo_ids() == ()
    with service._repository._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_objects "
                "WHERE object_type = 'undo' AND object_id = ?",
                (expiring_undo,),
            ).fetchone()[0]
            == 0
        )


def test_expired_working_context_is_filtered_but_not_deleted(service) -> None:
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    record = ProfileRecord(
        profile_id=manifest.profile_id,
        record_id="working-record",
        scope_id=scope.scope_id,
        kind="working_context",
        payload={"kind": "working_context", "subject": "task", "value": "draft"},
        semantic_key={"namespace": "working", "subject": "task"},
        state="active",
        controls={"sync_mode": "device_only", "agent_visibility": "agent_visible"},
        provenance={"source": "manual", "actor": "user", "reason_code": "edit"},
        version_id="working-v1",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    service.create_record(record)
    service.clock = lambda: NOW + timedelta(hours=2)

    assert service.list_records(scope_ids=(scope.scope_id,)) == ()
    assert service.get_record(record.record_id) == record

    replacement = ProfileRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "record_id": "working-replacement",
            "version_id": "working-v2",
            "created_at": NOW + timedelta(hours=2),
            "updated_at": NOW + timedelta(hours=2),
            "expires_at": NOW + timedelta(hours=3),
        }
    )
    assert service.create_record(replacement) == replacement


def test_unknown_profile_scope_and_invalid_restore_state_are_rejected(service) -> None:
    service.create_profile()
    record = _preference(service)
    with pytest.raises(ValueError, match="profile"):
        service.create_record(
            ProfileRecord.model_validate(
                {**record.model_dump(mode="python"), "profile_id": "foreign-profile"}
            )
        )
    with pytest.raises(ValueError, match="scope"):
        service.create_record(
            ProfileRecord.model_validate(
                {**record.model_dump(mode="python"), "scope_id": "missing-scope"}
            )
        )

    created = service.create_record(record)
    with pytest.raises(ValueError, match="archived"):
        service.restore_record(
            created.record_id, expected_version_id=created.version_id
        )


def test_archive_restore_preserve_user_only_device_only_controls(service) -> None:
    service.create_profile()
    source = _preference(service)
    private = ProfileRecord.model_validate(
        {
            **source.model_dump(mode="python"),
            "controls": {
                "sync_mode": "device_only",
                "agent_visibility": "user_only",
            },
        }
    )
    created = service.create_record(private)
    archived = service.archive_record(
        created.record_id, expected_version_id=created.version_id
    )
    restored = service.restore_record(
        created.record_id, expected_version_id=archived.version_id
    )
    assert restored.controls == private.controls


def test_undo_body_is_encrypted_and_expires_after_exact_24_hours(service) -> None:
    service.create_profile()
    source = _preference(service)
    first = service.create_record(
        ProfileRecord.model_validate(
            {
                **source.model_dump(mode="python"),
                "payload": {
                    "kind": "preference",
                    "subject": "response.detail",
                    "polarity": "like",
                    "value": "UNDO-BEFORE-CANARY",
                },
            }
        )
    )
    service.update_record(
        first.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value="UNDO-NEW-CANARY"
            )
        ),
        expected_version_id=first.version_id,
    )
    assert b"UNDO-BEFORE-CANARY" not in service._repository.db_path.read_bytes()
    assert service.list_undo_ids()
    service.clock = lambda: NOW + timedelta(hours=24)
    assert service.list_undo_ids() == ()


def test_syncable_to_device_only_update_splits_identity_without_private_outbox(
    service,
) -> None:
    service.create_profile()
    first = service.create_record(_preference(service))
    first = service.update_record(
        first.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail",
                polarity="like",
                value="PRE-SPLIT-CANARY",
            )
        ),
        expected_version_id=first.version_id,
    )
    stale_undo = service.list_undo_ids()[0]
    private_value = "DEVICE-ONLY-PRIVATE-CANARY"
    with service._repository._connect() as connection:
        prior_outbox = connection.execute(
            "SELECT outbox_id FROM encrypted_outbox "
            "WHERE object_type = 'record' AND object_id = ?",
            (first.record_id,),
        ).fetchall()
        connection.execute(
            "UPDATE encrypted_outbox SET status = 'sent' WHERE outbox_id = ?",
            (prior_outbox[0]["outbox_id"],),
        )

    converted = service.update_record(
        first.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value=private_value
            ),
            controls=ProfileControls(
                sync_mode=SyncMode.DEVICE_ONLY,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            ),
        ),
        expected_version_id=first.version_id,
    )

    tombstone = service.get_record(first.record_id)
    assert tombstone is not None
    assert tombstone.state is RecordState.DELETED
    assert tombstone.payload is None
    assert tombstone.controls.sync_mode is SyncMode.SYNCABLE
    assert converted.record_id != first.record_id
    assert converted.parent_version_id is None
    assert converted.controls.sync_mode is SyncMode.DEVICE_ONLY
    assert converted.payload.value == private_value
    assert (
        service._repository.get_record_derivation(converted.record_id)
        == first.record_id
    )
    assert service.get_manifest().revision == 3
    assert service.list_undo_ids() == ()

    with service._repository._connect() as connection:
        outbox_rows = connection.execute(
            "SELECT outbox_id, object_id, version_id FROM encrypted_outbox "
            "ORDER BY rowid"
        ).fetchall()
        old_record_versions = connection.execute(
            "SELECT version_id FROM encrypted_objects "
            "WHERE object_type = 'record' AND object_id = ?",
            (first.record_id,),
        ).fetchall()
        retired_outboxes = connection.execute(
            "SELECT COUNT(*) FROM encrypted_objects WHERE object_type = 'outbox' "
            f"AND object_id IN ({','.join('?' for _ in prior_outbox)})",
            tuple(row["outbox_id"] for row in prior_outbox),
        ).fetchone()[0]
        stale_undo_rows = connection.execute(
            "SELECT COUNT(*) FROM encrypted_objects "
            "WHERE object_type = 'undo' AND object_id = ?",
            (stale_undo,),
        ).fetchone()[0]
    assert {row["object_id"] for row in outbox_rows} == {first.record_id}
    assert len(outbox_rows) == 1
    assert converted.record_id not in {row["object_id"] for row in outbox_rows}
    assert [row["version_id"] for row in old_record_versions] == [tombstone.version_id]
    assert retired_outboxes == 0
    assert stale_undo_rows == 0
    tombstone_outbox = next(
        row for row in outbox_rows if row["version_id"] == tombstone.version_id
    )
    assert (
        service._repository.get_outbox_body(tombstone_outbox["outbox_id"])["record"][
            "payload"
        ]
        is None
    )
    durable = service._repository.db_path.read_bytes()
    assert private_value.encode() not in durable
    assert b"derived_from" not in durable


def test_syncable_to_device_only_conversion_rolls_back_every_artifact(
    service, monkeypatch
) -> None:
    service.create_profile()
    first = service.create_record(_preference(service))
    manifest = service.get_manifest()

    def fail_outbox(*_args, **_kwargs):
        raise RuntimeError("injected outbox failure")

    monkeypatch.setattr(service._repository, "_insert_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="injected"):
        service.update_record(
            first.record_id,
            RecordMutation(
                controls=ProfileControls(
                    sync_mode=SyncMode.DEVICE_ONLY,
                    agent_visibility=AgentVisibility.AGENT_VISIBLE,
                )
            ),
            expected_version_id=first.version_id,
        )

    assert service.get_record(first.record_id) == first
    assert service.get_manifest() == manifest
    assert service._repository.list_records() == [first]
    with service._repository._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM local_record_links").fetchone()[0]
            == 0
        )


def test_device_only_split_keeps_manifest_fence_from_before_collision_scan(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first_service = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    first_service.create_profile()
    shared = first_service.create_record(
        _preference(first_service, value="split-source", subject="split-source")
    )
    second_service = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    scan_complete = threading.Event()
    concurrent_commit_complete = threading.Event()
    original_scan = first_service._require_no_collision

    def scan_then_wait(*args, **kwargs):
        original_scan(*args, **kwargs)
        scan_complete.set()
        assert concurrent_commit_complete.wait(5)

    first_service._require_no_collision = scan_then_wait

    def mutate_manifest():
        assert scan_complete.wait(5)
        try:
            second_service.create_record(
                _preference(
                    second_service,
                    value="concurrent",
                    subject="non-colliding-key",
                )
            )
        finally:
            concurrent_commit_complete.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = executor.submit(mutate_manifest)
        split = executor.submit(
            first_service.update_record,
            shared.record_id,
            RecordMutation(
                controls=ProfileControls(
                    sync_mode=SyncMode.DEVICE_ONLY,
                    agent_visibility=AgentVisibility.AGENT_VISIBLE,
                )
            ),
            expected_version_id=shared.version_id,
        )
        concurrent.result(timeout=5)
        with pytest.raises(ProfileConflictError):
            split.result(timeout=5)

    assert first_service.get_record(shared.record_id) == shared
    assert first_service._repository.get_record_derivation(shared.record_id) is None


@pytest.mark.parametrize(
    "operation",
    ["create", "update", "archive", "restore", "delete", "undo"],
)
def test_workspace_user_mutations_require_mapping_and_succeed_after_mapping(
    service, operation
) -> None:
    manifest = service.create_profile()
    scope = ProfileScope(
        scope_id=f"unmapped-{operation}",
        profile_id=manifest.profile_id,
        kind=ScopeKind.WORKSPACE,
        version_id=f"unmapped-{operation}-version",
        created_at=NOW,
        updated_at=NOW,
    )
    service._repository.commit_scope(scope)
    record = ProfileRecord.model_validate(
        {
            **_preference(service, value=operation).model_dump(mode="python"),
            "scope_id": scope.scope_id,
        }
    )

    if operation == "create":

        def mutate():
            return service.create_record(record)

    else:
        if operation == "restore":
            record = ProfileRecord.model_validate(
                {**record.model_dump(mode="python"), "state": RecordState.ARCHIVED}
            )
        service._repository.commit_record_version(record, expected_version_id=None)
        if operation == "update":

            def mutate():
                return service.update_record(
                    record.record_id,
                    RecordMutation(payload=record.payload),
                    expected_version_id=record.version_id,
                )

        elif operation == "archive":

            def mutate():
                return service.archive_record(
                    record.record_id, expected_version_id=record.version_id
                )

        elif operation == "restore":

            def mutate():
                return service.restore_record(
                    record.record_id, expected_version_id=record.version_id
                )

        elif operation == "delete":

            def mutate():
                return service.delete_record(
                    record.record_id, expected_version_id=record.version_id
                )

        else:
            changed = ProfileRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "version_id": f"{record.version_id}-changed",
                    "parent_version_id": record.version_id,
                }
            )
            current_manifest = service.get_manifest()
            service._repository.commit_record_and_manifest(
                changed,
                service._next_manifest(current_manifest),
                expected_record_version=record.version_id,
                expected_manifest_version=current_manifest.current_version_id,
                undo_id="unmapped-undo",
                undo_body={
                    "version": 1,
                    "expected_head_version": changed.version_id,
                    "before_record": record.model_dump(mode="json"),
                    "expires_at": "2026-08-30T12:00:00.000Z",
                },
                undo_expires_at="2026-08-30T12:00:00.000Z",
            )

            def mutate():
                return service.undo("unmapped-undo")

    with pytest.raises(ValueError, match="mapped"):
        mutate()

    service.map_workspace_scope(f"local-{operation}", scope.scope_id)
    assert mutate() is not None


@pytest.mark.parametrize("foreign_field", ["profile_id", "scope_id"])
def test_mutations_revalidate_current_record_identity(
    service, monkeypatch, foreign_field
) -> None:
    service.create_profile()
    current = _preference(service)
    foreign = ProfileRecord.model_validate(
        {
            **current.model_dump(mode="python"),
            foreign_field: f"foreign-{foreign_field}",
        }
    )
    monkeypatch.setattr(service._repository, "get_record", lambda _record_id: foreign)

    with pytest.raises(ValueError, match="profile|scope"):
        service.archive_record(
            foreign.record_id, expected_version_id=foreign.version_id
        )


def test_standalone_removal_requires_only_copy_confirmation(service) -> None:
    service.create_profile()
    service.create_record(_preference(service))

    with pytest.raises(ValueError, match="only copy"):
        service.remove_local_profile(confirm_only_copy=False)
    assert service.list_records(scope_ids=(service.list_scopes()[0].scope_id,))

    service.remove_local_profile(confirm_only_copy=True)
    assert service.status().state == "removed"
    with sqlite3.connect(service._repository.db_path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM encrypted_objects").fetchone()[0]
            == 0
        )


def test_settings_snapshot_is_immutable_and_reads_local_policy(service) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]
    active = service.create_record(_preference(service, value="active"))
    archived = service.create_record(
        _preference(service, value="archived", subject="archived-subject")
    )
    archived = service.archive_record(
        archived.record_id, expected_version_id=archived.version_id
    )
    service.set_runtime_enabled(True)
    service.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)

    snapshot = service.settings_snapshot()

    assert snapshot.status.runtime_enabled is True
    assert snapshot.scopes[0].scope == scope
    assert snapshot.scopes[0].authority is AgentAuthority.READ_ONLY
    assert snapshot.scopes[0].label == "Global"
    assert snapshot.records == (active, archived)
    with pytest.raises(AttributeError):
        snapshot.records = ()


def test_settings_snapshot_repr_does_not_expose_scope_content(service) -> None:
    service.create_profile()
    scope = service.create_workspace_scope(
        "workspace-repr-canary", "scope-label-repr-canary"
    )
    scope_row = SettingsScopeSnapshot(
        scope=scope,
        label="scope-label-repr-canary",
        linked=True,
        authority=AgentAuthority.PROPOSE,
    )
    snapshot = PersonalContextSettingsSnapshot(
        status=service.status(), scopes=(scope_row,)
    )

    assert "scope-label-repr-canary" not in repr(scope_row)
    assert scope.scope_id not in repr(scope_row)
    assert "scope-label-repr-canary" not in repr(snapshot)
    assert scope.scope_id not in repr(snapshot)


def test_settings_snapshot_fails_closed_for_corrupt_scope_policy(service) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    with sqlite3.connect(service._repository.db_path) as connection:
        connection.execute(
            "UPDATE encrypted_objects SET ciphertext = ? "
            "WHERE object_type = 'runtime_policy' AND object_id = ?",
            (b"corrupt", scope.scope_id),
        )

    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.settings_snapshot()

    assert caught.value.reason_code == "agent_authority_denied"


def test_settings_snapshot_labels_unlinked_workspace_without_granting_access(
    service,
) -> None:
    service.create_profile()
    service.set_runtime_enabled(True)
    scope = service.create_workspace_scope("workspace-local", "Private Label")
    version = service._repository.get_scope_binding_version(scope.scope_id)
    assert version is not None
    with service._repository._connect() as connection:
        connection.execute(
            "DELETE FROM encrypted_objects WHERE object_type = 'scope_binding' "
            "AND object_id = ? AND version_id = ?",
            (scope.scope_id, version),
        )

    snapshot = service.settings_snapshot()

    scope_row = next(row for row in snapshot.scopes if row.scope == scope)
    assert scope_row.label == "Unlinked workspace"
    assert scope_row.linked is False
    assert scope_row.authority is AgentAuthority.PROPOSE
    with pytest.raises(PersonalContextAuthorityError, match="scope_unmapped"):
        service.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    with pytest.raises(PersonalContextAuthorityError, match="scope_unmapped"):
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)


def test_manual_record_creation_owns_identity_and_provenance(service) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]

    created = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "device_only", "agent_visibility": "user_only"},
    )

    assert created.record_id.startswith("record-")
    assert created.version_id.startswith("record-version-")
    assert created.profile_id == service.get_manifest().profile_id
    assert created.scope_id == scope.scope_id
    assert created.kind.value == "preference"
    assert created.provenance.model_dump(mode="json") == {
        "source": "manual",
        "actor": "user",
        "reason_code": "settings_edit",
        "source_references": [],
        "source_hashes": [],
        "derived_from_record_id": None,
    }


def test_confirmed_removal_is_truthful_and_start_fresh_is_explicit(service) -> None:
    service.create_profile()
    previous_profile_id = service.get_manifest().profile_id

    service.remove_local_profile(confirm_only_copy=True)

    removed = service.status()
    assert removed.state == "removed"
    assert removed.profile_present is False
    assert removed.locked is False
    with pytest.raises(ValueError, match="removed"):
        service.create_profile()

    fresh = service.start_fresh_profile()

    assert fresh.profile_id != previous_profile_id
    assert service.status().state == "disabled"
    assert len(service.list_scopes()) == 1


def test_start_fresh_retries_incomplete_key_deletion_and_rotates_keys(
    service, memory_protector, monkeypatch
) -> None:
    service.create_profile()
    old_keys = next(iter(memory_protector._materials.values()))
    original_delete = memory_protector.delete
    delete_calls = 0

    def fail_first_delete(profile_ref: str) -> None:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise ProfileLockedError("injected key deletion failure")
        original_delete(profile_ref)

    monkeypatch.setattr(memory_protector, "delete", fail_first_delete)
    with pytest.raises(ProfileLockedError, match="injected"):
        service.remove_local_profile(confirm_only_copy=True)
    assert service.status().state == "removed"

    service.start_fresh_profile()

    new_keys = next(iter(memory_protector._materials.values()))
    assert delete_calls == 2
    assert new_keys != old_keys


def test_destructive_lifecycle_serializes_delayed_removal_before_start_fresh(
    service, memory_protector, monkeypatch
) -> None:
    service.create_profile()
    delete_started = threading.Event()
    allow_delete = threading.Event()
    original_delete = memory_protector.delete

    def delayed_first_delete(profile_ref: str) -> None:
        if not delete_started.is_set():
            delete_started.set()
            assert allow_delete.wait(5)
        original_delete(profile_ref)

    monkeypatch.setattr(memory_protector, "delete", delayed_first_delete)
    with ThreadPoolExecutor(max_workers=2) as executor:
        removal = executor.submit(service.remove_local_profile, confirm_only_copy=True)
        assert delete_started.wait(5)
        fresh = executor.submit(service.start_fresh_profile)
        try:
            fresh.result(timeout=0.5)
        except TimeoutError:
            pass
        finally:
            allow_delete.set()
        removal.result(timeout=5)
        fresh.result(timeout=5)

    assert service.status().state == "disabled"
    assert not memory_protector.is_empty
    reopened = PersonalContextService(
        PersonalContextRepository(
            service._repository.db_path, key_protector=memory_protector
        ),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    assert reopened.status().state == "disabled"


def test_scope_authority_rejects_older_intent_that_completes_last(service) -> None:
    service.create_profile()
    scope = service.list_scopes()[0]
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    expected_policy_version_id = service.settings_snapshot().scopes[0].policy_version_id
    assert expected_policy_version_id is not None
    older_started = threading.Event()
    release_older = threading.Event()

    def delayed_older_intent() -> None:
        older_started.set()
        assert release_older.wait(5)
        service.set_scope_authority(
            scope.scope_id,
            AgentAuthority.DIRECT_WRITE,
            expected_policy_version_id=expected_policy_version_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(delayed_older_intent)
        assert older_started.wait(5)
        service.set_scope_authority(
            scope.scope_id,
            AgentAuthority.READ_ONLY,
            expected_policy_version_id=expected_policy_version_id,
        )
        release_older.set()
        with pytest.raises(ProfileConflictError):
            older.result(timeout=5)

    assert service.get_scope_authority(scope.scope_id) is AgentAuthority.READ_ONLY


def test_finish_secure_removal_retries_key_deletion_without_starting_fresh(
    service, memory_protector, monkeypatch
) -> None:
    service.create_profile()
    original_delete = memory_protector.delete
    delete_calls = 0

    def fail_first_delete(profile_ref: str) -> None:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise ProfileLockedError("injected key deletion failure")
        original_delete(profile_ref)

    monkeypatch.setattr(memory_protector, "delete", fail_first_delete)
    with pytest.raises(ProfileLockedError, match="injected"):
        service.remove_local_profile(confirm_only_copy=True)

    service.finish_secure_removal()
    service.finish_secure_removal()

    assert service.status().state == "removed"
    assert memory_protector.is_empty
    assert delete_calls == 3


def test_finish_secure_removal_survives_restart_after_failed_key_deletion(
    tmp_path, memory_protector, monkeypatch
) -> None:
    db_path = tmp_path / "personal-context.db"
    first = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    first.create_profile()
    original_delete = memory_protector.delete
    monkeypatch.setattr(
        memory_protector,
        "delete",
        lambda _profile_ref: (_ for _ in ()).throw(ProfileLockedError("injected")),
    )
    with pytest.raises(ProfileLockedError, match="injected"):
        first.remove_local_profile(confirm_only_copy=True)

    monkeypatch.setattr(memory_protector, "delete", original_delete)
    reopened = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    reopened.finish_secure_removal()

    assert reopened.status().state == "removed"
    assert memory_protector.is_empty


def test_destroyed_repository_reopens_removed_without_creating_keys(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    first.create_profile()
    first.remove_local_profile(confirm_only_copy=True)
    assert memory_protector.is_empty

    reopened = PersonalContextService(
        PersonalContextRepository(db_path, key_protector=memory_protector),
        clock=lambda: NOW,
        id_factory=Ids(),
    )

    assert reopened.status().state == "removed"
    assert memory_protector.is_empty
    reopened.start_fresh_profile()
    assert reopened.status().state == "disabled"
