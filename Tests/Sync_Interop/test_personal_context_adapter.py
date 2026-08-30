from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

import pytest
from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileManifest,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    ProposalOperation,
    ProposalState,
    RecordKind,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)
from tldw_profile_core.canonical import canonical_json_bytes
from tldw_profile_core.models import ActorType

from tldw_chatbook.Sync_Interop.personal_context_adapter import (
    PersonalContextSyncAdapter,
    PersonalContextSyncValidationError,
)
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    ProfileConflictError,
)
from tldw_chatbook.tldw_api import SyncV2Envelope


INTEGRITY_KEY = b"k" * 32
INTEGRITY_KEY_ID = "pc-key-1"


@dataclass
class _Entry:
    outbox_id: str = "outbox-1"
    object_type: str = "record"
    object_id: str = "record-1"
    version_id: str = "record-version-1"


@dataclass
class _RecordingService:
    calls: list[dict[str, object]] = field(default_factory=list)

    def apply_sync_object(self, **values: object) -> object:
        self.calls.append(values)
        return values["value"]


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _adapter() -> PersonalContextSyncAdapter:
    return PersonalContextSyncAdapter(
        integrity_key=INTEGRITY_KEY,
        integrity_key_id=INTEGRITY_KEY_ID,
    )


def _record() -> ProfileRecord:
    return ProfileRecord(
        profile_id="profile-1",
        record_id="record-1",
        scope_id="scope-global",
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise answers"
        ),
        semantic_key=SemanticKey(
            namespace="preference", subject="response.detail"
        ),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
        provenance=ProfileProvenance(
            source="manual", actor=ActorType.USER, reason_code="settings_edit"
        ),
        version_id="record-version-1",
        parent_version_id=None,
        created_at="2026-08-30T12:00:00Z",
        updated_at="2026-08-30T12:00:00Z",
    )


def _proposal(*, profile_id: str = "profile-1", scope_id: str = "scope-global"):
    proposed_record = _record().model_copy(
        update={"profile_id": profile_id, "scope_id": scope_id}
    )
    return ProfileProposal(
        proposal_id="proposal-1",
        profile_id=profile_id,
        scope_id=scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed_record,
        provenance=ProfileProvenance(
            source="agent",
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at="2026-08-30T12:00:00Z",
        expires_at="2026-11-28T12:00:00Z",
    )


def _whole_object_cases():
    manifest = ProfileManifest(
        profile_id="profile-1",
        revision=2,
        purge_generation=0,
        created_at="2026-08-30T12:00:00Z",
        updated_at="2026-08-30T12:01:00Z",
        current_version_id="manifest-version-2",
    )
    scope = ProfileScope(
        scope_id="scope-global",
        profile_id="profile-1",
        kind="global",
        version_id="scope-version-1",
        created_at="2026-08-30T12:00:00Z",
        updated_at="2026-08-30T12:00:00Z",
    )
    purge = {
        "schema_version": 1,
        "profile_id": "profile-1",
        "purge_generation": 1,
    }
    return (
        ("manifest", manifest.profile_id, None, "upsert", manifest),
        ("scope", scope.scope_id, scope.profile_id, "upsert", scope),
        ("record", "record-1", "scope-global", "upsert", _record()),
        ("proposal", "proposal-1", "scope-global", "upsert", _proposal()),
        ("purge", "profile-1", None, "tombstone", purge),
    )


def test_outbound_adapter_builds_exact_canonical_hmac_whole_object() -> None:
    record = _record()
    payload = record.model_dump(mode="json")

    envelope = _adapter().build_envelope(
        entry=_Entry(),
        body={"version": 1, "record": payload},
        dataset_id="dataset-1",
        device_id="device-1",
    )

    expected = hmac.new(
        INTEGRITY_KEY, canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()
    assert envelope.domain == "personal_context.record"
    assert envelope.object_id == record.record_id
    assert envelope.parent_id == record.scope_id
    assert envelope.payload == payload
    assert envelope.payload_hash == f"hmac-sha256-v1:{expected}"
    assert envelope.routing_metadata["integrity_key_id"] == INTEGRITY_KEY_ID
    assert envelope.client_envelope_id == "personal-context:outbox-1"


@pytest.mark.parametrize(
    ("object_type", "object_id", "parent_id", "operation", "value"),
    _whole_object_cases(),
)
def test_outbound_adapter_supports_all_canonical_domains(
    object_type, object_id, parent_id, operation, value
) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    envelope = _adapter().build_envelope(
        entry=_Entry(
            object_type=object_type,
            object_id=object_id,
            version_id=f"{object_type}-version-1",
        ),
        body={"version": 1, object_type: payload},
        dataset_id="dataset-1",
        device_id="device-1",
    )

    assert envelope.domain == f"personal_context.{object_type}"
    assert envelope.object_id == object_id
    assert envelope.parent_id == parent_id
    assert envelope.operation == operation
    assert envelope.payload == payload


def test_inbound_adapter_rejects_integrity_before_calling_service() -> None:
    record = _record()
    service = _RecordingService()
    envelope = SyncV2Envelope(
        client_envelope_id="remote-1",
        dataset_id="dataset-1",
        domain="personal_context.record",
        object_id=record.record_id,
        parent_id=record.scope_id,
        operation="upsert",
        adapter_version=1,
        schema_version=1,
        device_id="remote-device",
        payload=record.model_dump(mode="json"),
        payload_hash="hmac-sha256-v1:" + "0" * 64,
        encryption_policy="server_trusted_v1",
        routing_metadata={"integrity_key_id": INTEGRITY_KEY_ID},
    )

    with pytest.raises(
        PersonalContextSyncValidationError,
        match="personal_context_integrity_invalid",
    ):
        _adapter().apply_inbound(envelope, service=service)

    assert service.calls == []


def test_adapter_rejects_pending_proposal_with_device_only_record() -> None:
    proposal = _proposal()
    proposal = proposal.model_copy(
        update={
            "proposed_record": proposal.proposed_record.model_copy(
                update={
                    "controls": proposal.proposed_record.controls.model_copy(
                        update={"sync_mode": SyncMode.DEVICE_ONLY}
                    )
                }
            )
        }
    )
    payload = proposal.model_dump(mode="json")

    with pytest.raises(
        PersonalContextSyncValidationError,
        match="invalid_canonical_object",
    ):
        _adapter().build_envelope(
            entry=_Entry(
                object_type="proposal",
                object_id=proposal.proposal_id,
                version_id="proposal-version-1",
            ),
            body={"version": 1, "proposal": payload},
            dataset_id="dataset-1",
            device_id="device-1",
        )


def test_inbound_adapter_applies_only_through_personal_context_service() -> None:
    record = _record()
    outbound = _adapter().build_envelope(
        entry=_Entry(),
        body={"version": 1, "record": record.model_dump(mode="json")},
        dataset_id="dataset-1",
        device_id="remote-device",
    )
    service = _RecordingService()

    result = _adapter().apply_inbound(outbound, service=service)

    assert result == record
    assert service.calls == [
        {
            "domain": "personal_context.record",
            "value": record,
            "actor_type": "sync",
            "actor_id": "remote-device",
            "base_object_hash": None,
        }
    ]


def test_envelope_applier_routes_personal_context_to_owner_service() -> None:
    record = _record()
    adapter = _adapter()
    envelope = adapter.build_envelope(
        entry=_Entry(),
        body={"version": 1, "record": record.model_dump(mode="json")},
        dataset_id="dataset-1",
        device_id="remote-device",
    )
    service = _RecordingService()
    applier = SyncEnvelopeApplier(
        local_store=object(),
        personal_context_adapter=adapter,
        personal_context_service=service,
    )

    assert applier.apply(envelope) == {
        "status": "applied",
        "domain": "personal_context.record",
        "entity_id": "record-1",
    }
    assert len(service.calls) == 1


def test_inbound_adapter_uses_real_service_lineage_fence(tmp_path) -> None:
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "profile.db", key_protector=InMemoryProfileKeyProtector()
        ),
        id_factory=_Ids(),
    )
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    record = _record().model_copy(
        update={"profile_id": manifest.profile_id, "scope_id": scope.scope_id}
    )
    adapter = _adapter()
    envelope = adapter.build_envelope(
        entry=_Entry(object_id=record.record_id),
        body={"version": 1, "record": record.model_dump(mode="json")},
        dataset_id="dataset-1",
        device_id="remote-device",
    )

    assert adapter.apply_inbound(envelope, service=service) == record
    assert service.get_record(record.record_id) == record

    stale = record.model_copy(
        update={"version_id": "record-version-2", "parent_version_id": "wrong"}
    )
    stale_envelope = adapter.build_envelope(
        entry=_Entry(
            outbox_id="outbox-2",
            object_id=record.record_id,
            version_id=stale.version_id,
        ),
        body={"version": 1, "record": stale.model_dump(mode="json")},
        dataset_id="dataset-1",
        device_id="remote-device",
    )

    with pytest.raises(ProfileConflictError):
        adapter.apply_inbound(stale_envelope, service=service)


def test_inbound_terminal_proposal_replaces_pending_proposal(tmp_path) -> None:
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "profile.db", key_protector=InMemoryProfileKeyProtector()
        ),
        id_factory=_Ids(),
    )
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    pending = _proposal(profile_id=manifest.profile_id, scope_id=scope.scope_id)
    service._repository.commit_proposal(pending, enqueue_outbox=False)
    terminal = pending.model_copy(
        update={
            "state": ProposalState.REJECTED,
            "proposed_record": None,
            "confidence": None,
        }
    )
    adapter = _adapter()
    envelope = adapter.build_envelope(
        entry=_Entry(
            object_type="proposal",
            object_id=terminal.proposal_id,
            version_id="proposal-version-terminal",
        ),
        body={"version": 1, "proposal": terminal.model_dump(mode="json")},
        dataset_id="dataset-1",
        device_id="remote-device",
    )

    assert adapter.apply_inbound(envelope, service=service) == terminal
    assert service._repository.get_proposal(terminal.proposal_id) == terminal


def test_sync_service_rejects_immutable_scope_and_record_rewrites(tmp_path) -> None:
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "profile.db", key_protector=InMemoryProfileKeyProtector()
        ),
        id_factory=_Ids(),
    )
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    record = _record().model_copy(
        update={"profile_id": manifest.profile_id, "scope_id": scope.scope_id}
    )
    service.apply_sync_object(
        domain="personal_context.record",
        value=record,
        actor_type="sync",
        actor_id="device-a",
    )
    rewritten_record = record.model_copy(
        update={
            "version_id": "record-version-2",
            "parent_version_id": record.version_id,
            "created_at": record.created_at.replace(day=29),
        }
    )

    with pytest.raises(ProfileConflictError):
        service.apply_sync_object(
            domain="personal_context.record",
            value=rewritten_record,
            actor_type="sync",
            actor_id="device-b",
        )

    rewritten_scope = scope.model_copy(
        update={"kind": ScopeKind.WORKSPACE, "version_id": "scope-version-2"}
    )
    with pytest.raises(ProfileConflictError):
        service.apply_sync_object(
            domain="personal_context.scope",
            value=rewritten_scope,
            actor_type="sync",
            actor_id="device-b",
        )
