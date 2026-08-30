from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    ProvenanceSource,
    RecordKind,
    RecordState,
    SemanticKey,
    SyncMode,
)
from tldw_profile_core.models import ActorType

from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_chatbook.Personal_Context.sync_outbox import ProfileSyncOutbox
from tldw_chatbook.Sync_Interop.personal_context_adapter import (
    PersonalContextSyncAdapter,
)
from tldw_chatbook.Sync_Interop.personal_context_dispatcher import (
    PersonalContextOutboxDispatcher,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
SCOPE = {
    "server_profile_id": "server-profile-1",
    "authenticated_principal_id": "user-1",
    "workspace_scope": None,
    "dataset_id": "dataset-1",
}
STORAGE_KEY = b"s" * 32


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _dependencies(tmp_path):
    profile_repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=InMemoryProfileKeyProtector()
    )
    service = PersonalContextService(
        profile_repository, clock=lambda: NOW, id_factory=_Ids()
    )
    service.create_profile()
    profile_outbox = ProfileSyncOutbox(profile_repository)
    sync_repository = SyncStateRepository(tmp_path / "sync.db")
    adapter = PersonalContextSyncAdapter(
        integrity_key=b"i" * 32,
        integrity_key_id="pc-key-1",
    )
    return profile_outbox, sync_repository, adapter, service


def _dispatcher(profile_outbox, sync_repository, adapter):
    return PersonalContextOutboxDispatcher(
        profile_outbox=profile_outbox,
        state_repository=sync_repository,
        adapter=adapter,
    )


def _record(profile_id: str, *, value: str = "concise answers") -> ProfileRecord:
    return ProfileRecord(
        profile_id=profile_id,
        record_id="record-1",
        scope_id="scope-global",
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value=value
        ),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
        provenance=ProfileProvenance(
            source=ProvenanceSource.MANUAL,
            actor=ActorType.USER,
            reason_code="settings_edit",
        ),
        version_id="record-version-1",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )


def test_dispatcher_crash_between_databases_replays_without_duplicate(
    tmp_path, monkeypatch
) -> None:
    profile_outbox, sync_repository, adapter, _service = _dependencies(tmp_path)
    expected = len(profile_outbox.list_pending())
    dispatcher = _dispatcher(profile_outbox, sync_repository, adapter)

    def crash_once(*_args, **_kwargs):
        raise RuntimeError("injected cross-database crash")

    monkeypatch.setattr(dispatcher, "_after_destination_enqueue", crash_once)
    with pytest.raises(RuntimeError, match="cross-database crash"):
        dispatcher.dispatch_pending(
            device_id="device-1", storage_key=STORAGE_KEY, **SCOPE
        )

    first_pass = sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE)
    assert len(first_pass) == 1
    assert profile_outbox.list_pending()

    result = _dispatcher(
        profile_outbox, sync_repository, adapter
    ).dispatch_pending(device_id="device-1", storage_key=STORAGE_KEY, **SCOPE)

    pending = sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE)
    assert result == {"dispatched": expected, "quarantined": 0}
    assert len(pending) == expected
    assert len({entry["client_envelope_id"] for entry in pending}) == expected
    assert profile_outbox.list_pending() == ()


def test_crash_recovery_preserves_source_when_staged_ciphertext_is_invalid(
    tmp_path, monkeypatch
) -> None:
    profile_outbox, sync_repository, adapter, _service = _dependencies(tmp_path)
    entry = profile_outbox.list_pending(limit=1)[0]
    dispatcher = _dispatcher(profile_outbox, sync_repository, adapter)

    def crash_once(*_args, **_kwargs):
        raise RuntimeError("injected cross-database crash")

    monkeypatch.setattr(dispatcher, "_after_destination_enqueue", crash_once)
    with pytest.raises(RuntimeError, match="cross-database crash"):
        dispatcher.dispatch_pending(
            device_id="device-1",
            storage_key=STORAGE_KEY,
            limit=1,
            **SCOPE,
        )

    existing = sync_repository.get_sync_v2_outbox_entry(
        client_envelope_id=f"personal-context:{entry.outbox_id}",
        **SCOPE,
    )
    assert existing is not None
    corrupted = dict(existing["envelope"])
    corrupted["payload_ciphertext"] = "corrupt"
    with sync_repository._get_connection() as connection:
        connection.execute(
            "UPDATE sync_v2_local_outbox SET envelope = ? "
            "WHERE client_envelope_id = ?",
            (
                json.dumps(corrupted),
                f"personal-context:{entry.outbox_id}",
            ),
        )

    result = _dispatcher(
        profile_outbox, sync_repository, adapter
    ).dispatch_pending(
        device_id="device-1",
        storage_key=STORAGE_KEY,
        limit=1,
        **SCOPE,
    )

    assert result == {"dispatched": 0, "quarantined": 1}
    assert profile_outbox.get_quarantine_reason(entry.outbox_id) == (
        "destination_copy_invalid"
    )
    assert profile_outbox.read_body(entry.outbox_id) is not None


def test_crash_recovery_preserves_source_when_staged_metadata_is_tampered(
    tmp_path,
    monkeypatch,
) -> None:
    profile_outbox, sync_repository, adapter, _service = _dependencies(tmp_path)
    entry = profile_outbox.list_pending(limit=1)[0]
    dispatcher = _dispatcher(profile_outbox, sync_repository, adapter)

    def crash_once(*_args, **_kwargs):
        raise RuntimeError("injected cross-database crash")

    monkeypatch.setattr(dispatcher, "_after_destination_enqueue", crash_once)
    with pytest.raises(RuntimeError, match="cross-database crash"):
        dispatcher.dispatch_pending(
            device_id="device-1",
            storage_key=STORAGE_KEY,
            limit=1,
            **SCOPE,
        )

    existing = sync_repository.get_sync_v2_outbox_entry(
        client_envelope_id=f"personal-context:{entry.outbox_id}",
        **SCOPE,
    )
    assert existing is not None
    tampered = dict(existing["envelope"])
    tampered["deleted"] = not tampered["deleted"]
    with sync_repository._get_connection() as connection:
        connection.execute(
            "UPDATE sync_v2_local_outbox SET envelope = ? "
            "WHERE client_envelope_id = ?",
            (
                json.dumps(tampered),
                f"personal-context:{entry.outbox_id}",
            ),
        )

    result = _dispatcher(
        profile_outbox, sync_repository, adapter
    ).dispatch_pending(
        device_id="device-1",
        storage_key=STORAGE_KEY,
        limit=1,
        **SCOPE,
    )

    assert result == {"dispatched": 0, "quarantined": 1}
    assert profile_outbox.get_quarantine_reason(entry.outbox_id) == (
        "destination_copy_invalid"
    )
    assert profile_outbox.read_body(entry.outbox_id) is not None


def test_dispatcher_quarantines_poisoned_body_without_copying_content(
    tmp_path,
) -> None:
    profile_outbox, sync_repository, adapter, _service = _dependencies(tmp_path)
    for entry in profile_outbox.list_pending():
        profile_outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    poison_id = profile_outbox.repository.commit_outbox_body(
        object_type="record",
        object_id="record-poison",
        version_id="version-poison",
        body={"version": 1, "record": {"secret": "OUTBOX-POISON-CANARY"}},
    )

    result = _dispatcher(
        profile_outbox, sync_repository, adapter
    ).dispatch_pending(device_id="device-1", storage_key=STORAGE_KEY, **SCOPE)

    assert result == {"dispatched": 0, "quarantined": 1}
    assert profile_outbox.get_quarantine_reason(poison_id) == (
        "invalid_canonical_object"
    )
    assert sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE) == []
    assert b"OUTBOX-POISON-CANARY" not in sync_repository.db_path.read_bytes()


def test_dispatcher_quarantines_unauthenticated_encrypted_body(tmp_path) -> None:
    profile_outbox, sync_repository, adapter, _service = _dependencies(tmp_path)
    entry = profile_outbox.list_pending(limit=1)[0]
    with profile_outbox.repository._transaction() as connection:
        envelope_version = connection.execute(
            "SELECT envelope_version FROM encrypted_outbox WHERE outbox_id = ?",
            (entry.outbox_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE encrypted_objects SET ciphertext = ? "
            "WHERE object_type = 'outbox' AND object_id = ? AND version_id = ?",
            (b"corrupt", entry.outbox_id, envelope_version),
        )

    result = _dispatcher(
        profile_outbox, sync_repository, adapter
    ).dispatch_pending(
        device_id="device-1", storage_key=STORAGE_KEY, limit=1, **SCOPE
    )

    assert result == {"dispatched": 0, "quarantined": 1}
    assert profile_outbox.get_quarantine_reason(entry.outbox_id) == (
        "encrypted_body_unavailable"
    )
    assert sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE) == []


def test_successful_dispatch_keeps_profile_canary_encrypted_in_sync_state(
    tmp_path,
) -> None:
    profile_outbox, sync_repository, adapter, service = _dependencies(tmp_path)
    for entry in profile_outbox.list_pending():
        profile_outbox.acknowledge(entry.outbox_id, f"bootstrap:{entry.outbox_id}")
    manifest = service.get_manifest()
    record = _record(
        manifest.profile_id,
        value="SYNC-STAGING-PLAINTEXT-CANARY-9f61",
    ).model_copy(update={"scope_id": service.list_scopes()[0].scope_id})
    service.create_record(record)

    result = _dispatcher(
        profile_outbox, sync_repository, adapter
    ).dispatch_pending(device_id="device-1", storage_key=STORAGE_KEY, **SCOPE)

    assert result["dispatched"] == 2
    entries = sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE)
    record_entry = next(
        entry for entry in entries if entry["domain"] == "personal_context.record"
    )
    stored = SyncV2Envelope.model_validate(record_entry["envelope"])
    assert stored.payload == {}
    assert stored.payload_ciphertext
    restored = adapter.restore_from_storage(stored, storage_key=STORAGE_KEY)
    assert (
        restored.payload["payload"]["value"]
        == "SYNC-STAGING-PLAINTEXT-CANARY-9f61"
    )
    durable = b"".join(
        path.read_bytes()
        for path in tmp_path.iterdir()
        if path.name.startswith("sync.db")
    )
    assert b"SYNC-STAGING-PLAINTEXT-CANARY-9f61" not in durable


def test_dispatcher_binds_next_version_to_acknowledged_sync_head(
    tmp_path,
) -> None:
    profile_outbox, sync_repository, adapter, service = _dependencies(tmp_path)
    _dispatcher(profile_outbox, sync_repository, adapter).dispatch_pending(
        device_id="device-1", storage_key=STORAGE_KEY, **SCOPE
    )
    bootstrap = sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE)
    accepted = []
    for cursor, entry in enumerate(bootstrap, start=10):
        envelope = SyncV2Envelope.model_validate(entry["envelope"])
        accepted.append(
            {
                "client_envelope_id": envelope.client_envelope_id,
                "server_sequence": cursor,
                "object_revision": envelope.object_revision,
            }
        )
    sync_repository.mark_sync_v2_outbox_push_results(
        accepted=accepted,
        rejected=[],
        conflicts=[],
        **SCOPE,
    )
    manifest = service.get_manifest()
    record = _record(manifest.profile_id).model_copy(
        update={"scope_id": service.list_scopes()[0].scope_id}
    )
    service.create_record(record)

    _dispatcher(profile_outbox, sync_repository, adapter).dispatch_pending(
        device_id="device-1", storage_key=STORAGE_KEY, **SCOPE
    )

    pending = sync_repository.list_pending_sync_v2_outbox_envelopes(**SCOPE)
    manifest_envelope = next(
        adapter.restore_from_storage(
            SyncV2Envelope.model_validate(entry["envelope"]),
            storage_key=STORAGE_KEY,
        )
        for entry in pending
        if entry["domain"] == "personal_context.manifest"
    )
    prior_manifest = next(
        SyncV2Envelope.model_validate(entry["envelope"])
        for entry in bootstrap
        if entry["domain"] == "personal_context.manifest"
    )
    prior_receipt = next(
        item
        for item in accepted
        if item["client_envelope_id"] == prior_manifest.client_envelope_id
    )
    assert manifest_envelope.base_server_cursor == prior_receipt["server_sequence"]
    assert manifest_envelope.base_object_revision == prior_manifest.object_revision
    assert manifest_envelope.base_object_hash == prior_manifest.payload_hash
