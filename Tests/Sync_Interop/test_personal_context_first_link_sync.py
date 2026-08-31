from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileManifest,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.reconciliation import (
    CanonicalBootstrapSnapshot,
    build_reconciliation_plan,
    canonical_snapshot_heads,
)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_chatbook.Sync_Interop.personal_context_adapter import (
    PersonalContextSyncAdapter,
)
from tldw_chatbook.Sync_Interop.personal_context_first_link_sync import (
    PersonalContextFirstLinkSync,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope


class _Dispatcher:
    adapter = object()

    def __init__(self, gate=None) -> None:
        self.calls = []
        self.gate = gate

    def dispatch_first_link_reconciliation(self, **kwargs):
        if self.gate is not None:
            assert self.gate.active_plan == "plan-1"
        self.calls.append(kwargs)
        return {"dispatched": 0, "quarantined": 0}


class _FreezeGate:
    def __init__(self) -> None:
        self.active_plan = None
        self.entered_plans = []

    @contextmanager
    def first_link_reconciliation_writes(self, *, plan_id):
        assert self.active_plan is None
        self.active_plan = plan_id
        self.entered_plans.append(plan_id)
        try:
            yield
        finally:
            self.active_plan = None


class _ReconciliationProfile(_FreezeGate):
    def __init__(self) -> None:
        super().__init__()
        self.applied = []

    def append(self, value):
        self.applied.append(value)


class _Server:
    def __init__(self) -> None:
        self.calls = []

    async def pull_v2_envelopes(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "dataset_id": kwargs["dataset_id"],
            "envelopes": [],
            "next_cursor": kwargs["cursor"],
            "has_more": False,
        }

    async def _pull_v2_personal_context_first_link(self, **kwargs):
        return await self.pull_v2_envelopes(**kwargs)


@pytest.mark.asyncio
async def test_special_first_link_pull_uses_existing_transport_cursor_and_includes_own(
    tmp_path,
):
    state = SyncStateRepository(tmp_path / "sync.db")
    scope = {
        "server_profile_id": "server-1",
        "authenticated_principal_id": "user-1",
    }
    heads = {"personal_context.manifest": {"profile-1": "manifest-v1"}}
    state.set_sync_v2_profile_state(
        **scope,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "17"},
    )
    state.set_personal_context_link_state(
        **scope,
        state="reconciling",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="personal-context-bootstrap-v1:" + "a" * 64,
        bootstrap_heads=heads,
        expected_heads=heads,
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )
    gate = _FreezeGate()
    dispatcher = _Dispatcher(gate)
    server = _Server()
    sync = PersonalContextFirstLinkSync(
        server_service=server,
        state_repository=state,
        dispatcher=dispatcher,
        personal_context_service=gate,
        local_store=object(),
        dataset_keys={"dataset-1": b"s" * 32},
    )

    result = await sync.converge(
        **scope,
        device_id="device-1",
        dataset_id="dataset-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="personal-context-bootstrap-v1:" + "a" * 64,
        bootstrap_heads=heads,
        expected_heads=heads,
    )

    assert result == {
        "confirmed_cursor": "17",
        "confirmed_heads": heads,
    }
    assert server.calls == [
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "cursor": "17",
            "domains": [
                "personal_context.manifest",
                "personal_context.scope",
                "personal_context.record",
                "personal_context.proposal",
                "personal_context.purge",
            ],
            "page_size": None,
            "include_own_changes": True,
        }
    ]
    assert len(dispatcher.calls) == 1
    assert gate.entered_plans == ["plan-1"]

    sync._profile = object()
    with pytest.raises(
        RuntimeError, match="personal_context_reconciliation_gate_unavailable"
    ):
        await sync.converge(
            **scope,
            device_id="device-1",
            dataset_id="dataset-1",
            profile_id="profile-1",
            integrity_key_id="key-1",
            key_record_id="key-record-1",
            purge_generation=0,
            bootstrap_cursor="personal-context-bootstrap-v1:" + "a" * 64,
            bootstrap_heads=heads,
            expected_heads=heads,
        )
    assert len(dispatcher.calls) == 1


@pytest.mark.asyncio
async def test_server_only_unbound_workspace_is_outside_reviewed_convergence_heads(
    tmp_path,
):
    state = SyncStateRepository(tmp_path / "sync.db")
    scope = {
        "server_profile_id": "server-1",
        "authenticated_principal_id": "user-1",
    }
    bootstrap = {
        "personal_context.manifest": {"profile-1": "manifest-v1"},
        "personal_context.scope": {
            "scope-global": "scope-global-v1",
            "scope-unbound": "scope-unbound-v1",
        },
        "personal_context.record": {"record-unbound": "record-unbound-v1"},
        "personal_context.proposal": {},
    }
    expected = {
        "personal_context.manifest": {"profile-1": "manifest-v1"},
        "personal_context.scope": {"scope-global": "scope-global-v1"},
        "personal_context.record": {},
        "personal_context.proposal": {},
    }
    state.set_sync_v2_profile_state(
        **scope,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "cursor-bootstrap"},
    )
    state.set_personal_context_link_state(
        **scope,
        state="reconciling",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=bootstrap,
        expected_heads=expected,
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )
    profile = _FreezeGate()
    sync = PersonalContextFirstLinkSync(
        server_service=_Server(),
        state_repository=state,
        dispatcher=_Dispatcher(),
        personal_context_service=profile,
        local_store=object(),
        dataset_keys={"dataset-1": b"s" * 32},
    )

    result = await sync.converge(
        **scope,
        device_id="device-1",
        dataset_id="dataset-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=bootstrap,
        expected_heads=expected,
    )

    assert result == {
        "confirmed_cursor": None,
        "confirmed_heads": expected,
    }


class _EchoAdapter:
    @staticmethod
    def restore_from_storage(envelope, *, storage_key):
        assert storage_key == b"s" * 32
        return envelope

    @staticmethod
    def apply_inbound(_envelope, *, service):
        service.append("applied")


class _DeltaDispatcher(_Dispatcher):
    adapter = _EchoAdapter()


class _DeltaServer(_Server):
    def __init__(self, envelope) -> None:
        super().__init__()
        self.envelope = envelope
        self.pushes = []

    async def push_v2_envelopes(self, **kwargs):
        self.pushes.append(kwargs)
        return {
            "dataset_id": kwargs["dataset_id"],
            "accepted": [
                {"client_envelope_id": self.envelope.client_envelope_id}
            ],
            "rejected": [],
            "conflicts": [],
        }

    async def _push_v2_personal_context_first_link(self, **kwargs):
        return await self.push_v2_envelopes(**kwargs)

    async def pull_v2_envelopes(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "dataset_id": kwargs["dataset_id"],
            "envelopes": [self.envelope.model_dump(mode="json")],
            "next_cursor": "cursor-confirmed",
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_unreviewed_remote_version_is_rejected_before_privileged_apply(tmp_path):
    state = SyncStateRepository(tmp_path / "sync.db")
    scope = {
        "server_profile_id": "server-1",
        "authenticated_principal_id": "user-1",
    }
    reviewed = {
        "personal_context.manifest": {"profile-1": "manifest-v1"},
        "personal_context.scope": {"scope-1": "scope-v1"},
        "personal_context.record": {},
    }
    state.set_sync_v2_profile_state(
        **scope,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={},
    )
    state.set_personal_context_link_state(
        **scope,
        state="reconciling",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=reviewed,
        expected_heads=reviewed,
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )
    concurrent = SyncV2Envelope(
        client_envelope_id="other-device:late",
        dataset_id="dataset-1",
        domain="personal_context.record",
        object_id="record-late",
        parent_id="scope-1",
        operation="upsert",
        device_id="other-device",
        base_version=None,
        entity_version="record-late-v1",
        payload={"schema_version": 1},
        payload_hash="hmac-sha256-v1:" + "b" * 64,
        encryption_policy="server_trusted_v1",
    )
    server = _DeltaServer(concurrent)
    profile = _ReconciliationProfile()
    sync = PersonalContextFirstLinkSync(
        server_service=server,
        state_repository=state,
        dispatcher=_DeltaDispatcher(),
        personal_context_service=profile,
        local_store=object(),
        dataset_keys={"dataset-1": b"s" * 32},
    )

    with pytest.raises(
        RuntimeError, match="personal_context_reviewed_lineage_changed"
    ):
        await sync.converge(
            **scope,
            device_id="device-1",
            dataset_id="dataset-1",
            profile_id="profile-1",
            integrity_key_id="key-1",
            key_record_id="key-record-1",
            purge_generation=0,
            bootstrap_cursor="cursor-bootstrap",
            bootstrap_heads=reviewed,
            expected_heads=reviewed,
        )

    assert profile.applied == []


@pytest.mark.asyncio
async def test_concurrent_remote_object_cannot_mutate_production_repository(tmp_path):
    now = "2026-08-30T12:00:00.000Z"
    manifest = ProfileManifest(
        profile_id="profile-1",
        revision=1,
        purge_generation=0,
        created_at=now,
        updated_at=now,
        current_version_id="manifest-v1",
    )
    scope_model = ProfileScope(
        profile_id="profile-1",
        scope_id="scope-1",
        kind=ScopeKind.GLOBAL,
        version_id="scope-v1",
        created_at=now,
        updated_at=now,
    )
    repository = PersonalContextRepository(
        tmp_path / "profile.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    repository.create_profile_with_global_scope(manifest, scope_model)
    profile = PersonalContextService(repository)
    reviewed_snapshot = CanonicalBootstrapSnapshot(
        dataset_id="dataset-1",
        authority_id="authority-1",
        manifest=manifest,
        scopes=(scope_model,),
        records=(),
        proposals=(),
        purge_generation=0,
        schema_version=1,
        quotas={"max_record_bytes": 16_384},
        cursor="cursor-bootstrap",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        wrapped_key_blob="content-free-test-wrapper",
    )
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(scope_model,),
        local_records=(),
        local_proposals=(),
        remote=reviewed_snapshot,
        local_workspace_bindings={},
    )
    repository.acquire_first_link_freeze(
        plan_id=plan.plan_id,
        snapshot_token=plan.local_snapshot_token,
    )
    concurrent_record = ProfileRecord(
        profile_id="profile-1",
        record_id="record-late",
        scope_id="scope-1",
        kind="preference",
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="unreviewed"
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
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id="record-late-v1",
        parent_version_id=None,
        created_at=now,
        updated_at=now,
    )
    adapter = PersonalContextSyncAdapter(
        integrity_key=b"k" * 32,
        integrity_key_id="key-1",
    )
    concurrent = adapter.build_envelope(
        entry=SimpleNamespace(
            outbox_id="remote-late",
            object_type="record",
            object_id="record-late",
            version_id="record-late-v1",
        ),
        body={"version": 1, "record": concurrent_record.model_dump(mode="json")},
        dataset_id="dataset-1",
        device_id="other-device",
    )
    state = SyncStateRepository(tmp_path / "sync.db")
    scope = {
        "server_profile_id": "server-1",
        "authenticated_principal_id": "user-1",
    }
    reviewed = canonical_snapshot_heads(manifest, (scope_model,), (), ())
    state.set_sync_v2_profile_state(
        **scope,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={},
    )
    state.set_personal_context_link_state(
        **scope,
        state="reconciling",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=reviewed,
        expected_heads=reviewed,
        plan_id=plan.plan_id,
        rebaseline_version=1,
        attention_code=None,
    )
    dispatcher = _Dispatcher()
    dispatcher.adapter = adapter
    sync = PersonalContextFirstLinkSync(
        server_service=_DeltaServer(concurrent),
        state_repository=state,
        dispatcher=dispatcher,
        personal_context_service=profile,
        local_store=object(),
        dataset_keys={"dataset-1": b"s" * 32},
    )

    with pytest.raises(
        RuntimeError, match="personal_context_reviewed_lineage_changed"
    ):
        await sync.converge(
            **scope,
            device_id="device-1",
            dataset_id="dataset-1",
            profile_id="profile-1",
            integrity_key_id="key-1",
            key_record_id="key-record-1",
            purge_generation=0,
            bootstrap_cursor="cursor-bootstrap",
            bootstrap_heads=reviewed,
            expected_heads=reviewed,
        )

    assert repository.list_records() == []
    repository.release_first_link_freeze(plan_id=plan.plan_id)
    changed_snapshot = CanonicalBootstrapSnapshot(
        dataset_id=reviewed_snapshot.dataset_id,
        authority_id=reviewed_snapshot.authority_id,
        manifest=reviewed_snapshot.manifest,
        scopes=reviewed_snapshot.scopes,
        records=(concurrent_record,),
        proposals=reviewed_snapshot.proposals,
        purge_generation=reviewed_snapshot.purge_generation,
        schema_version=reviewed_snapshot.schema_version,
        quotas=reviewed_snapshot.quotas,
        cursor="cursor-after-concurrent-write",
        integrity_key_id=reviewed_snapshot.integrity_key_id,
        key_record_id=reviewed_snapshot.key_record_id,
        wrapped_key_blob=reviewed_snapshot.wrapped_key_blob,
    )
    next_plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(scope_model,),
        local_records=(),
        local_proposals=(),
        remote=changed_snapshot,
        local_workspace_bindings={},
    )
    assert next_plan.remote_only_record_ids == ("record-late",)


@pytest.mark.asyncio
async def test_special_cycle_requires_echoed_materialized_delta_before_confirming(tmp_path):
    state = SyncStateRepository(tmp_path / "sync.db")
    scope = {
        "server_profile_id": "server-1",
        "authenticated_principal_id": "user-1",
    }
    bootstrap = {"personal_context.manifest": {"profile-1": "manifest-v1"}}
    expected = {
        **bootstrap,
        "personal_context.record": {"record-1": "record-v2"},
    }
    state.set_sync_v2_profile_state(
        **scope,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "19"},
    )
    state.set_personal_context_link_state(
        **scope,
        state="reconciling",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=bootstrap,
        expected_heads=expected,
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )
    envelope = SyncV2Envelope(
        client_envelope_id="personal-context:1",
        dataset_id="dataset-1",
        domain="personal_context.record",
        object_id="record-1",
        parent_id="scope-1",
        operation="upsert",
        device_id="device-1",
        base_version=None,
        entity_version="record-v2",
        payload={"schema_version": 1},
        payload_hash="hmac-sha256-v1:" + "a" * 64,
        encryption_policy="server_trusted_v1",
    )
    state.enqueue_sync_v2_outbox_envelope(
        **scope,
        workspace_scope=None,
        dataset_id="dataset-1",
        envelope=envelope,
    )
    server = _DeltaServer(envelope)
    profile = _ReconciliationProfile()
    sync = PersonalContextFirstLinkSync(
        server_service=server,
        state_repository=state,
        dispatcher=_DeltaDispatcher(),
        personal_context_service=profile,
        local_store=object(),
        dataset_keys={"dataset-1": b"s" * 32},
    )

    result = await sync.converge(
        **scope,
        device_id="device-1",
        dataset_id="dataset-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=bootstrap,
        expected_heads=expected,
    )

    assert result["confirmed_heads"] == expected
    assert result["confirmed_cursor"] == "cursor-confirmed"
    assert len(server.pushes) == 1
    assert server.pushes[0]["last_known_cursor"] == "19"
    assert server.pushes[0]["last_known_cursor"] != "cursor-bootstrap"
    assert server.calls[0]["include_own_changes"] is True
    assert server.calls[0]["cursor"] == "19"
    assert profile.applied == ["applied"]


class _PagedDeltaDispatcher:
    adapter = _EchoAdapter()

    def __init__(self, state, scope, envelopes) -> None:
        self.state = state
        self.scope = scope
        self.remaining = list(envelopes)
        self.limits = []

    def dispatch_first_link_reconciliation(self, **kwargs):
        limit = int(kwargs["limit"])
        self.limits.append(limit)
        page = self.remaining[:limit]
        del self.remaining[:limit]
        for envelope in page:
            self.state.enqueue_sync_v2_outbox_envelope(
                **self.scope,
                workspace_scope=None,
                dataset_id=kwargs["dataset_id"],
                envelope=envelope,
            )
        return {"dispatched": len(page), "quarantined": 0}


class _PagedDeltaServer(_Server):
    def __init__(self, envelopes) -> None:
        super().__init__()
        self.envelopes = list(envelopes)
        self.pushes = []

    async def push_v2_envelopes(self, **kwargs):
        self.pushes.append(kwargs)
        return {
            "dataset_id": kwargs["dataset_id"],
            "accepted": [
                {"client_envelope_id": item["client_envelope_id"]}
                for item in kwargs["envelopes"]
            ],
            "rejected": [],
            "conflicts": [],
        }

    async def _push_v2_personal_context_first_link(self, **kwargs):
        return await self.push_v2_envelopes(**kwargs)

    async def pull_v2_envelopes(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "dataset_id": kwargs["dataset_id"],
            "envelopes": [item.model_dump(mode="json") for item in self.envelopes],
            "next_cursor": "cursor-confirmed-101",
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_special_cycle_drains_101_entries_in_negotiated_push_batches(tmp_path):
    state = SyncStateRepository(tmp_path / "sync.db")
    scope = {
        "server_profile_id": "server-1",
        "authenticated_principal_id": "user-1",
    }
    bootstrap = {"personal_context.manifest": {"profile-1": "manifest-v1"}}
    envelopes = [
        SyncV2Envelope(
            client_envelope_id=f"personal-context:{index:03d}",
            dataset_id="dataset-1",
            domain="personal_context.record",
            object_id=f"record-{index:03d}",
            parent_id="scope-1",
            operation="upsert",
            device_id="device-1",
            base_version=None,
            entity_version=f"record-v{index:03d}",
            payload={"schema_version": 1},
            payload_hash="hmac-sha256-v1:" + f"{index:064x}",
            encryption_policy="server_trusted_v1",
        )
        for index in range(101)
    ]
    expected = {
        **bootstrap,
        "personal_context.record": {
            str(item.object_id): str(item.entity_version) for item in envelopes
        },
    }
    state.set_sync_v2_profile_state(
        **scope,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "cursor-bootstrap"},
        capabilities={"max_batch_size": 40},
    )
    state.set_personal_context_link_state(
        **scope,
        state="reconciling",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=bootstrap,
        expected_heads=expected,
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )
    dispatcher = _PagedDeltaDispatcher(state, scope, envelopes)
    server = _PagedDeltaServer(envelopes)
    profile = _ReconciliationProfile()
    sync = PersonalContextFirstLinkSync(
        server_service=server,
        state_repository=state,
        dispatcher=dispatcher,
        personal_context_service=profile,
        local_store=object(),
        dataset_keys={"dataset-1": b"s" * 32},
    )

    result = await sync.converge(
        **scope,
        device_id="device-1",
        dataset_id="dataset-1",
        profile_id="profile-1",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="cursor-bootstrap",
        bootstrap_heads=bootstrap,
        expected_heads=expected,
    )

    assert result["confirmed_heads"] == expected
    assert [len(call["envelopes"]) for call in server.pushes] == [40, 40, 21]
    assert dispatcher.limits[:3] == [40, 40, 40]
    assert dispatcher.remaining == []
    assert len(profile.applied) == 101
