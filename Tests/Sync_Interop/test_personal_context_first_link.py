from __future__ import annotations

import pytest
from tldw_profile_core import ProfileManifest, ProfileScope, ScopeKind

from tldw_chatbook.Personal_Context import link_service as link_service_module
from tldw_chatbook.Personal_Context.link_key_custody import (
    InMemoryPersonalContextLinkKeyCustodian,
)
from tldw_chatbook.Personal_Context.link_service import PersonalContextLinkService
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.reconciliation import (
    CanonicalBootstrapSnapshot,
    build_reconciliation_plan,
)
from tldw_chatbook.Personal_Context.repository import (
    PersonalContextRepository,
    ProfileKeyActivationPendingError,
    release_first_link_freeze_for_recovery,
)
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop.sync_readiness import (
    PERSONAL_CONTEXT_MINIMUM_QUOTAS,
)
from tldw_chatbook.tldw_api import SyncV2Envelope
from tldw_chatbook.tldw_api.exceptions import (
    PersonalContextBootstrapAttentionError,
)
from tldw_chatbook.tldw_api.sync_schemas import (
    SyncPersonalContextPurgeAttention,
    SyncPersonalContextQuotaAttention,
    SyncPersonalContextSchemaAttention,
)


SCOPE = {
    "server_profile_id": "server-config-1",
    "authenticated_principal_id": "user-1",
}


def test_link_state_persists_exact_binding_and_gates_ordinary_sync(tmp_path) -> None:
    path = tmp_path / "sync.db"
    repo = SyncStateRepository(path)

    repo.set_personal_context_link_state(
        **SCOPE,
        state="local_rebaseline_complete",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=2,
        bootstrap_cursor="sha256:" + "b" * 64,
        plan_id="plan-1",
        rebaseline_version=1,
        attention_code=None,
    )
    repo.close()

    reopened = SyncStateRepository(path)
    state = reopened.get_personal_context_link_state(**SCOPE)

    assert state["state"] == "local_rebaseline_complete"
    assert state["profile_id"] == "profile-server"
    assert state["integrity_key_id"] == "key-1"
    assert state["purge_generation"] == 2
    assert reopened.personal_context_sync_enabled(**SCOPE) is False

    reopened.set_personal_context_link_state(
        **SCOPE,
        state="complete",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=2,
        bootstrap_cursor="sha256:" + "b" * 64,
        confirmed_cursor="cursor-confirmed",
        plan_id="plan-1",
        rebaseline_version=1,
        attention_code=None,
    )
    reopened.set_sync_v2_profile_state(
        **SCOPE,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "cursor-confirmed"},
    )
    assert reopened.personal_context_sync_enabled(
        **SCOPE,
        dataset_id="dataset-1",
        device_id="device-1",
        profile_id="profile-server",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=2,
        confirmed_cursor="cursor-confirmed",
    ) is True


def test_cancelled_plan_removes_only_unapproved_link_state(tmp_path) -> None:
    repo = SyncStateRepository(tmp_path / "sync.db")
    repo.set_personal_context_link_state(
        **SCOPE,
        state="review_required",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="sha256:" + "c" * 64,
        plan_id="plan-1",
        rebaseline_version=1,
        attention_code=None,
    )

    assert repo.cancel_personal_context_link_plan(**SCOPE, plan_id="plan-1") is True
    assert repo.get_personal_context_link_state(**SCOPE) is None
    assert repo.cancel_personal_context_link_plan(**SCOPE, plan_id="plan-1") is False


@pytest.mark.parametrize("state", ["applying", "local_rebaseline_complete"])
def test_cancel_refuses_after_canonical_apply_may_have_started(tmp_path, state) -> None:
    repo = SyncStateRepository(tmp_path / "sync.db")
    repo.set_personal_context_link_state(
        **SCOPE,
        state=state,
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=0,
        bootstrap_cursor="sha256:" + "d" * 64,
        plan_id="plan-1",
        rebaseline_version=1,
        attention_code=None,
    )

    with pytest.raises(ValueError, match="cannot_cancel"):
        repo.cancel_personal_context_link_plan(**SCOPE, plan_id="plan-1")


def _replace_plan(repo, *, state: str, plan_id: str) -> None:
    repo.set_personal_context_link_state(
        **SCOPE,
        state=state,
        device_id=f"device-{plan_id}",
        dataset_id=f"dataset-{plan_id}",
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id=f"key-{plan_id}",
        key_record_id=f"key-record-{plan_id}",
        purge_generation=0,
        bootstrap_cursor="sha256:" + "f" * 64,
        plan_id=plan_id,
        rebaseline_version=1,
        attention_code=("server_snapshot_stale" if state == "attention_required" else None),
    )


def test_retry_can_replace_an_unapproved_attention_snapshot(tmp_path) -> None:
    repo = SyncStateRepository(tmp_path / "sync.db")
    _replace_plan(repo, state="attention_required", plan_id="plan-1")

    _replace_plan(repo, state="attention_required", plan_id="plan-2")

    assert repo.get_personal_context_link_state(**SCOPE)["plan_id"] == "plan-2"


def test_new_review_cannot_overwrite_an_apply_in_progress(tmp_path) -> None:
    repo = SyncStateRepository(tmp_path / "sync.db")
    _replace_plan(repo, state="applying", plan_id="plan-1")

    with pytest.raises(ValueError, match="state_stale"):
        _replace_plan(repo, state="review_required", plan_id="plan-2")


NOW = "2026-08-30T12:00:00.000Z"


def _manifest(profile_id: str, version_id: str) -> ProfileManifest:
    return ProfileManifest(
        profile_id=profile_id,
        revision=0,
        purge_generation=0,
        created_at=NOW,
        updated_at=NOW,
        current_version_id=version_id,
    )


def _scope(profile_id: str, scope_id: str) -> ProfileScope:
    return ProfileScope(
        profile_id=profile_id,
        scope_id=scope_id,
        kind=ScopeKind.GLOBAL,
        version_id=f"{scope_id}-version",
        created_at=NOW,
        updated_at=NOW,
    )


class FakeProfileService:
    def __init__(self) -> None:
        self.manifest = _manifest("profile-local", "manifest-local")
        self.scopes = (_scope("profile-local", "scope-local"),)
        self.apply_calls = []
        self.activation_pending = False
        self.frozen_plan_id = None

    def first_link_snapshot(self):
        return self.manifest, self.scopes, (), (), {}

    def get_manifest(self):
        return self.manifest

    def apply_reviewed_link(self, **kwargs):
        self.apply_calls.append(kwargs)
        if self.activation_pending:
            raise ProfileKeyActivationPendingError(
                "injected secure-custody activation interruption"
            )
        self.manifest = kwargs["remote"].manifest
        return {"rebaseline_version": 2}

    def first_link_sync_heads(self):
        return {
            "personal_context.manifest": {
                self.manifest.profile_id: self.manifest.current_version_id
            }
        }

    def acquire_first_link_freeze(self, *, plan_id, snapshot_token):
        assert snapshot_token
        self.frozen_plan_id = plan_id

    def release_first_link_freeze(self, *, plan_id):
        if self.frozen_plan_id != plan_id:
            return False
        self.frozen_plan_id = None
        return True

    def first_link_apply_recovery_state(self, **_kwargs):
        if self.manifest.profile_id == "profile-local":
            return ("uncommitted", None)
        return ("committed", 2)


class FakeServerSync:
    def __init__(self, *, max_batch_size: int | None = None) -> None:
        self.bootstrap_calls = []
        self.complete_calls = []
        self.fail_complete_once = False
        self.max_batch_size = max_batch_size

    async def bootstrap_personal_context_link(self, **kwargs):
        self.bootstrap_calls.append(kwargs)
        manifest = _manifest("profile-server", "manifest-server")
        scope = _scope("profile-server", "scope-server")
        response = {
            "device_id": "device-1",
            "dataset_id": "dataset-1",
            "authority_id": "authority-1",
            "manifest": manifest.model_dump(mode="json"),
            "scopes": [scope.model_dump(mode="json")],
            "records": [],
            "proposals": [],
            "purge_generation": 0,
            "schema_version": 1,
            "quotas": dict(PERSONAL_CONTEXT_MINIMUM_QUOTAS),
            "cursor": "sha256:" + "e" * 64,
            "integrity_key_id": "integrity-1",
            "key_record_id": "key-record-1",
            "wrapped_key_blob": "wrapped",
        }
        if self.max_batch_size is not None:
            response["_sync_capabilities"] = {
                "max_batch_size": self.max_batch_size
            }
        return response

    async def complete_personal_context_link(self, **kwargs):
        self.complete_calls.append(kwargs)
        if self.fail_complete_once:
            self.fail_complete_once = False
            raise RuntimeError("temporary failure")


class AttentionServerSync(FakeServerSync):
    def __init__(self, attention) -> None:
        super().__init__()
        self.attention = attention

    async def bootstrap_personal_context_link(self, **kwargs):
        self.bootstrap_calls.append(kwargs)
        raise PersonalContextBootstrapAttentionError(self.attention)


class FakeFirstLinkSync:
    def __init__(self) -> None:
        self.calls = []
        self.fail_once = False
        self.fail_unconfirmed = False
        self.failure: Exception | None = None

    async def converge(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("partial transfer")
        if self.fail_unconfirmed:
            raise RuntimeError("personal_context_convergence_unconfirmed")
        return {
            "confirmed_cursor": "cursor-confirmed-9",
            "confirmed_heads": kwargs["expected_heads"],
        }


class FakeWrappingProvider:
    public_key_pem = "public"

    def unwrap_integrity_key(self, blob, *, integrity_key_id):
        assert blob == "wrapped"
        assert integrity_key_id == "integrity-1"
        return b"s" * 32


class _FailingWrappingProvider(FakeWrappingProvider):
    def unwrap_integrity_key(self, blob, *, integrity_key_id):
        raise ValueError("wrapped_integrity_key_invalid")


class _StageFailsCustodian(InMemoryPersonalContextLinkKeyCustodian):
    def stage(self, *, integrity_key: bytes, **binding: str) -> None:
        raise RuntimeError("secure stage unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attention",
    (
        SyncPersonalContextSchemaAttention(
            kind="schema_incompatible",
            required_schema_version=3,
            server_min_schema_version=1,
            server_max_schema_version=2,
        ),
        SyncPersonalContextQuotaAttention(
            kind="quota_incompatible",
            required_quotas={"max_record_bytes": 16_384},
            available_quotas={"max_record_bytes": 8_192},
            insufficient_quotas=["max_record_bytes"],
        ),
        SyncPersonalContextPurgeAttention(
            kind="purge_generation_mismatch",
            expected_purge_generation=1,
            current_purge_generation=2,
        ),
    ),
)
async def test_plan_maps_typed_bootstrap_attention_without_creating_review_state(
    tmp_path,
    attention,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=AttentionServerSync(attention),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    attention_error_type = getattr(
        link_service_module, "PersonalContextLinkAttentionRequired"
    )

    with pytest.raises(attention_error_type) as caught:
        await service.plan()

    assert caught.value.attention is attention
    assert str(caught.value) == "personal_context_link_attention_required"
    assert state.get_personal_context_link_state(**SCOPE) is None
    assert profile.frozen_plan_id is None


@pytest.mark.asyncio
async def test_plan_is_read_only_and_apply_resumes_after_complete_interruption(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    server = FakeServerSync()
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )

    plan = await service.plan()

    assert profile.apply_calls == []
    assert server.complete_calls == []
    assert state.get_personal_context_link_state(**SCOPE)["state"] == "review_required"

    server.fail_complete_once = True
    with pytest.raises(RuntimeError, match="temporary failure"):
        await service.apply(plan.plan_id, {})

    assert len(profile.apply_calls) == 1
    assert state.get_personal_context_link_state(**SCOPE)["state"] == (
        "local_rebaseline_complete"
    )
    receipt = await service.resume()

    assert len(profile.apply_calls) == 1
    assert len(server.complete_calls) == 2
    assert receipt.profile_id == "profile-server"
    complete = state.get_personal_context_link_state(**SCOPE)
    assert state.personal_context_sync_enabled(
        **SCOPE,
        **{
            key: complete[key]
            for key in (
                "dataset_id",
                "device_id",
                "profile_id",
                "integrity_key_id",
                "key_record_id",
                "purge_generation",
                "confirmed_cursor",
            )
        },
    ) is True


@pytest.mark.asyncio
async def test_restart_discards_only_expired_review_before_freezing_fresh_plan(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    server = FakeServerSync()
    state = SyncStateRepository(tmp_path / "sync.db")
    first = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    expired = await first.plan()
    assert profile.frozen_plan_id == expired.plan_id

    restarted = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    fresh = await restarted.plan()

    assert fresh.plan_id != expired.plan_id
    assert profile.frozen_plan_id == fresh.plan_id
    assert state.get_personal_context_link_state(**SCOPE)["plan_id"] == fresh.plan_id
    assert profile.apply_calls == []
    assert len(server.bootstrap_calls) == 2


@pytest.mark.asyncio
async def test_restart_releases_orphaned_persisted_freeze_without_link_state(
    tmp_path,
) -> None:
    protector = InMemoryProfileKeyProtector()
    profile_path = tmp_path / "profile.db"
    sync_path = tmp_path / "sync.db"
    repository = PersonalContextRepository(
        profile_path,
        key_protector=protector,
    )
    profile = PersonalContextService(repository)
    profile.create_profile()
    server = FakeServerSync()
    response = await server.bootstrap_personal_context_link()
    remote = CanonicalBootstrapSnapshot.from_response(response)
    manifest, scopes, records, proposals, bindings = profile.first_link_snapshot()
    orphan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=scopes,
        local_records=records,
        local_proposals=proposals,
        remote=remote,
        local_workspace_bindings=bindings,
    )
    profile.acquire_first_link_freeze(
        plan_id=orphan.plan_id,
        snapshot_token=orphan.local_snapshot_token,
    )
    initial_state = SyncStateRepository(sync_path)
    assert initial_state.get_personal_context_link_state(**SCOPE) is None
    initial_state.close()

    restarted_profile = PersonalContextService(
        PersonalContextRepository(profile_path, key_protector=protector)
    )
    restarted_state = SyncStateRepository(sync_path)
    restarted = PersonalContextLinkService(
        personal_context_service=restarted_profile,
        server_sync_service=server,
        state_repository=restarted_state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )

    fresh = await restarted.plan()

    assert fresh.plan_id != orphan.plan_id
    assert restarted_state.get_personal_context_link_state(**SCOPE)["plan_id"] == (
        fresh.plan_id
    )
    assert restarted.cancel(fresh.plan_id) is True


@pytest.mark.asyncio
async def test_locked_restart_releases_only_the_exact_content_free_freeze(
    tmp_path,
) -> None:
    profile_path = tmp_path / "profile.db"
    repository = PersonalContextRepository(
        profile_path,
        key_protector=InMemoryProfileKeyProtector(),
    )
    profile = PersonalContextService(repository)
    profile.create_profile()
    response = await FakeServerSync().bootstrap_personal_context_link()
    remote = CanonicalBootstrapSnapshot.from_response(response)
    manifest, scopes, records, proposals, bindings = profile.first_link_snapshot()
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=scopes,
        local_records=records,
        local_proposals=proposals,
        remote=remote,
        local_workspace_bindings=bindings,
    )
    profile.acquire_first_link_freeze(
        plan_id=plan.plan_id,
        snapshot_token=plan.local_snapshot_token,
    )

    assert (
        release_first_link_freeze_for_recovery(
            profile_path, plan_id="different-plan"
        )
        is False
    )
    assert profile.first_link_freeze_plan_id() == plan.plan_id
    assert (
        release_first_link_freeze_for_recovery(profile_path, plan_id=plan.plan_id)
        is True
    )
    assert profile.first_link_freeze_plan_id() is None


class _DeleteFailsOnceCustodian(InMemoryPersonalContextLinkKeyCustodian):
    def __init__(self) -> None:
        super().__init__()
        self.fail_delete = True

    def delete(self, **binding: str) -> None:
        if self.fail_delete:
            self.fail_delete = False
            raise RuntimeError("secure cleanup unavailable")
        super().delete(**binding)


@pytest.mark.asyncio
async def test_complete_receipt_preserves_freeze_until_staged_key_cleanup_retry(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    custodian = _DeleteFailsOnceCustodian()
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(RuntimeError, match="secure cleanup unavailable"):
        await service.apply(plan.plan_id, {})

    complete = state.get_personal_context_link_state(**SCOPE)
    assert complete["state"] == "complete"
    assert profile.frozen_plan_id == plan.plan_id
    receipt = await service.resume()
    assert receipt.confirmed_cursor == "cursor-confirmed-9"
    assert profile.frozen_plan_id is None


@pytest.mark.asyncio
async def test_applying_recovery_clears_stale_pc_destination_before_convergence(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    await service.plan()
    review = state.get_personal_context_link_state(**SCOPE)
    state.set_personal_context_link_state(
        **SCOPE,
        **{
            key: review[key]
            for key in (
                "device_id",
                "dataset_id",
                "authority_id",
                "profile_id",
                "integrity_key_id",
                "key_record_id",
                "purge_generation",
                "bootstrap_cursor",
                "plan_id",
                "rebaseline_version",
                "bootstrap_heads",
                "expected_heads",
            )
        },
        state="applying",
        attention_code=None,
        expected_states=("review_required",),
    )
    stale = SyncV2Envelope(
        client_envelope_id="stale-before-crash",
        dataset_id="dataset-1",
        domain="personal_context.record",
        object_id="record-stale",
        parent_id=None,
        operation="upsert",
        device_id="device-1",
        base_version=None,
        entity_version="record-stale-v1",
        payload={"schema_version": 1},
        payload_hash="hmac-sha256-v1:" + "b" * 64,
        encryption_policy="server_trusted_v1",
    )
    state.enqueue_sync_v2_outbox_envelope(
        **SCOPE,
        workspace_scope=None,
        dataset_id="dataset-1",
        envelope=stale,
    )

    await service.resume_after_local_activation(rebaseline_version=2)

    assert state.list_pending_sync_v2_outbox_envelopes(
        **SCOPE,
        workspace_scope=None,
        dataset_id="dataset-1",
        domains=["personal_context.record"],
    ) == []


@pytest.mark.asyncio
async def test_server_complete_stays_gated_until_exact_convergence_is_confirmed(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    server = FakeServerSync()
    convergence = FakeFirstLinkSync()
    convergence.fail_once = True
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=convergence,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(RuntimeError, match="partial transfer"):
        await service.apply(plan.plan_id, {})

    interrupted = state.get_personal_context_link_state(**SCOPE)
    assert interrupted["state"] == "reconciling"
    assert interrupted["confirmed_cursor"] is None
    assert state.personal_context_sync_enabled(**SCOPE) is False
    assert len(server.complete_calls) == 1

    receipt = await service.resume()

    assert len(server.complete_calls) == 1
    assert len(convergence.calls) == 2
    assert receipt.confirmed_cursor == "cursor-confirmed-9"
    complete = state.get_personal_context_link_state(**SCOPE)
    assert complete["state"] == "complete"
    assert complete["confirmed_cursor"] == "cursor-confirmed-9"


@pytest.mark.asyncio
async def test_changed_server_heads_return_to_attention_and_release_review_freeze(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    server = FakeServerSync()
    convergence = FakeFirstLinkSync()
    convergence.fail_unconfirmed = True
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=convergence,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(
        RuntimeError, match="personal_context_convergence_unconfirmed"
    ):
        await service.apply(plan.plan_id, {})

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == "server_snapshot_changed"
    assert attention["confirmed_cursor"] is None
    assert profile.frozen_plan_id is None
    assert state.personal_context_sync_enabled(**SCOPE) is False


@pytest.mark.asyncio
async def test_terminal_attention_cleanup_failure_does_not_mask_convergence_error(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    convergence = FakeFirstLinkSync()
    convergence.fail_unconfirmed = True
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=_DeleteFailsOnceCustodian(),
        first_link_sync=convergence,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(
        RuntimeError, match="personal_context_convergence_unconfirmed"
    ):
        await service.apply(plan.plan_id, {})

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == "server_snapshot_changed"
    assert profile.frozen_plan_id == plan.plan_id
    retry = await service.plan()
    assert retry.plan_id != plan.plan_id


@pytest.mark.parametrize(
    ("failure", "attention_code"),
    (
        (
            RuntimeError("personal_context_reconciliation_push_rejected"),
            "reconciliation_push_rejected",
        ),
        (
            RuntimeError("personal_context_reconciliation_apply_failed"),
            "reconciliation_apply_failed",
        ),
        (
            RuntimeError("personal_context_reconciliation_version_missing"),
            "reconciliation_version_missing",
        ),
        (
            ValueError("personal_context_reconciliation_binding_stale"),
            "reconciliation_binding_stale",
        ),
        (
            ValueError("Sync v2 push response omitted submitted client_envelope_id"),
            "reconciliation_validation_failed",
        ),
    ),
)
@pytest.mark.asyncio
async def test_terminal_reconciliation_failures_release_freeze_for_replan(
    tmp_path, failure, attention_code
) -> None:
    profile = FakeProfileService()
    convergence = FakeFirstLinkSync()
    convergence.failure = failure
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=convergence,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(type(failure), match=str(failure)):
        await service.apply(plan.plan_id, {})

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == attention_code
    assert profile.frozen_plan_id is None
    assert state.personal_context_sync_enabled(**SCOPE) is False


@pytest.mark.asyncio
async def test_missing_persisted_staging_key_releases_freeze_for_replan(tmp_path) -> None:
    profile = FakeProfileService()
    convergence = FakeFirstLinkSync()
    convergence.fail_once = True
    state = SyncStateRepository(tmp_path / "sync.db")
    custodian = InMemoryPersonalContextLinkKeyCustodian()
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        first_link_sync=convergence,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()
    with pytest.raises(RuntimeError, match="partial transfer"):
        await service.apply(plan.plan_id, {})
    custodian._storage.clear()

    with pytest.raises(ValueError, match="personal_context_staging_key_unavailable"):
        await service.resume()

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == "staging_key_unavailable"
    assert profile.frozen_plan_id is None


@pytest.mark.asyncio
async def test_sync_profile_binding_conflict_releases_freeze_for_replan(tmp_path) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()
    state.set_sync_v2_profile_state(
        **SCOPE,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="other-device",
        dataset_id="other-dataset",
        dataset_cursors={"sync_v2": plan.bootstrap_cursor},
    )

    with pytest.raises(RuntimeError, match="sync_profile_binding_conflict"):
        await service.apply(plan.plan_id, {})

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == "sync_profile_binding_conflict"
    assert profile.frozen_plan_id is None


@pytest.mark.asyncio
async def test_fresh_review_removes_only_stale_pending_pc_destination_copies(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    first_failure = FakeFirstLinkSync()
    first_failure.failure = RuntimeError(
        "personal_context_reconciliation_push_rejected"
    )
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=first_failure,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    failed_plan = await service.plan()
    with pytest.raises(RuntimeError, match="push_rejected"):
        await service.apply(failed_plan.plan_id, {})

    def envelope(envelope_id: str, domain: str, device_id: str) -> SyncV2Envelope:
        return SyncV2Envelope(
            client_envelope_id=envelope_id,
            dataset_id="dataset-1",
            domain=domain,
            object_id=envelope_id,
            parent_id=None,
            operation="upsert",
            device_id=device_id,
            base_version=None,
            entity_version="version-1",
            payload={"schema_version": 1},
            payload_hash="hmac-sha256-v1:" + "a" * 64,
            encryption_policy="server_trusted_v1",
        )

    stale_pc = envelope(
        "stale-pc", "personal_context.record", "device-1"
    )
    other_domain = envelope("keep-notes", "notes", "device-1")
    other_device = envelope(
        "keep-other-device", "personal_context.record", "device-2"
    )
    for item in (stale_pc, other_domain, other_device):
        state.enqueue_sync_v2_outbox_envelope(
            **SCOPE,
            workspace_scope=None,
            dataset_id="dataset-1",
            envelope=item,
        )

    service._first_link_sync = FakeFirstLinkSync()
    fresh_plan = await service.plan()
    await service.apply(fresh_plan.plan_id, {})

    pending = state.list_pending_sync_v2_outbox_envelopes(
        **SCOPE,
        workspace_scope=None,
        dataset_id="dataset-1",
    )
    assert {item["client_envelope_id"] for item in pending} == {
        "keep-notes",
        "keep-other-device",
    }


def test_complete_gate_requires_the_exact_link_and_sync_profile_binding(tmp_path) -> None:
    state = SyncStateRepository(tmp_path / "sync.db")
    state.set_sync_v2_profile_state(
        **SCOPE,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "cursor-confirmed-9"},
        capabilities={"personal_context": {"schema_version": 1}},
    )
    state.set_personal_context_link_state(
        **SCOPE,
        state="complete",
        device_id="device-1",
        dataset_id="dataset-1",
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        purge_generation=2,
        bootstrap_cursor="cursor-bootstrap-3",
        confirmed_cursor="cursor-confirmed-9",
        expected_heads={
            "personal_context.manifest": {"profile-server": "manifest-server"}
        },
        plan_id="plan-1",
        rebaseline_version=2,
        attention_code=None,
    )

    exact = {
        **SCOPE,
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-server",
        "integrity_key_id": "key-1",
        "key_record_id": "key-record-1",
        "purge_generation": 2,
        "confirmed_cursor": "cursor-confirmed-9",
    }
    assert state.personal_context_sync_enabled(**exact) is True
    assert state.personal_context_sync_enabled(
        **{**exact, "dataset_id": "dataset-stale"}
    ) is False
    assert state.personal_context_sync_enabled(
        **{**exact, "device_id": "device-stale"}
    ) is False
    assert state.personal_context_sync_enabled(
        **{**exact, "integrity_key_id": "key-stale"}
    ) is False
    assert state.personal_context_sync_enabled(**SCOPE) is False
    for omitted in tuple(exact.keys() - SCOPE.keys()):
        incomplete = dict(exact)
        incomplete.pop(omitted)
        assert state.personal_context_sync_enabled(**incomplete) is False


@pytest.mark.asyncio
async def test_plan_seeds_normal_sync_profile_without_generic_enrollment(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    server = FakeServerSync(max_batch_size=17)
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )

    await service.plan()

    seeded = state.get_sync_v2_profile_state(**SCOPE, workspace_scope=None)
    assert seeded["device_id"] == "device-1"
    assert seeded["dataset_id"] == "dataset-1"
    assert seeded["dataset_cursors"] == {}
    assert seeded["capabilities"]["personal_context"]["schema_version"] == 1
    assert seeded["capabilities"]["max_batch_size"] == 17


@pytest.mark.asyncio
async def test_plan_merges_matching_existing_sync_profile_state(tmp_path) -> None:
    state = SyncStateRepository(tmp_path / "sync.db")
    state.set_sync_v2_profile_state(
        **SCOPE,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "generic-cursor", "notes": "notes-cursor"},
        capabilities={"supported_domains": ["notes"], "max_batch_size": 50},
        dry_run_metadata={"existing": True},
    )
    service = PersonalContextLinkService(
        personal_context_service=FakeProfileService(),
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )

    await service.plan()

    seeded = state.get_sync_v2_profile_state(**SCOPE, workspace_scope=None)
    assert seeded["dataset_cursors"] == {
        "sync_v2": "generic-cursor",
        "notes": "notes-cursor",
    }
    assert seeded["capabilities"]["max_batch_size"] == 50
    assert seeded["capabilities"]["supported_domains"] == [
        "notes",
        "personal_context.manifest",
        "personal_context.scope",
        "personal_context.record",
        "personal_context.proposal",
        "personal_context.purge",
    ]
    assert seeded["dry_run_metadata"] == {"existing": True}


@pytest.mark.asyncio
async def test_plan_refuses_to_replace_existing_dataset_or_device_binding(tmp_path) -> None:
    state = SyncStateRepository(tmp_path / "sync.db")
    state.set_sync_v2_profile_state(
        **SCOPE,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="other-device",
        dataset_id="other-dataset",
        dataset_cursors={"sync_v2": "generic-cursor"},
        capabilities={"supported_domains": ["notes"]},
    )
    service = PersonalContextLinkService(
        personal_context_service=FakeProfileService(),
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )

    with pytest.raises(ValueError, match="sync_profile_binding_conflict"):
        await service.plan()

    unchanged = state.get_sync_v2_profile_state(**SCOPE, workspace_scope=None)
    assert unchanged["device_id"] == "other-device"
    assert unchanged["dataset_id"] == "other-dataset"


@pytest.mark.asyncio
async def test_activation_pending_keeps_staged_key_and_applying_gate(tmp_path) -> None:
    profile = FakeProfileService()
    profile.activation_pending = True
    server = FakeServerSync()
    state = SyncStateRepository(tmp_path / "sync.db")
    custodian = InMemoryPersonalContextLinkKeyCustodian()
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=server,
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(ProfileKeyActivationPendingError):
        await service.apply(plan.plan_id, {})

    interrupted = state.get_personal_context_link_state(**SCOPE)
    assert interrupted["state"] == "applying"
    assert state.personal_context_sync_enabled(**SCOPE) is False
    assert custodian.load(**service._key_binding(interrupted)) == b"s" * 32
    assert server.complete_calls == []

    receipt = await service.resume_after_local_activation(rebaseline_version=2)

    assert receipt.rebaseline_version == 2
    complete = state.get_personal_context_link_state(**SCOPE)
    assert state.personal_context_sync_enabled(
        **SCOPE,
        **{
            key: complete[key]
            for key in (
                "dataset_id",
                "device_id",
                "profile_id",
                "integrity_key_id",
                "key_record_id",
                "purge_generation",
                "confirmed_cursor",
            )
        },
    ) is True
    with pytest.raises(ValueError, match="binding_mismatch"):
        custodian.load(**service._key_binding(interrupted))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrapping", "custodian", "message", "attention_code"),
    (
        (
            _FailingWrappingProvider(),
            InMemoryPersonalContextLinkKeyCustodian(),
            "wrapped_integrity_key_invalid",
            "local_key_unwrap_failed",
        ),
        (
            FakeWrappingProvider(),
            _StageFailsCustodian(),
            "secure stage unavailable",
            "local_key_stage_failed",
        ),
    ),
)
async def test_key_preparation_failure_releases_freeze_and_enters_attention(
    tmp_path,
    wrapping,
    custodian,
    message,
    attention_code,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=wrapping,
        key_custodian=custodian,
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises((RuntimeError, ValueError), match=message):
        await service.apply(plan.plan_id, {})

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == attention_code
    assert profile.frozen_plan_id is None


@pytest.mark.asyncio
async def test_apply_failure_is_not_masked_when_staged_key_cleanup_also_fails(
    tmp_path,
) -> None:
    profile = FakeProfileService()

    def fail_apply(**_kwargs):
        raise RuntimeError("canonical apply failed")

    profile.apply_reviewed_link = fail_apply
    state = SyncStateRepository(tmp_path / "sync.db")
    custodian = _DeleteFailsOnceCustodian()
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(RuntimeError, match="canonical apply failed"):
        await service.apply(plan.plan_id, {})

    attention = state.get_personal_context_link_state(**SCOPE)
    assert attention["state"] == "attention_required"
    assert attention["attention_code"] == "local_apply_failed"
    assert profile.frozen_plan_id == plan.plan_id
    retry = await service.plan()
    assert retry.plan_id != plan.plan_id


@pytest.mark.asyncio
async def test_post_rebaseline_auxiliary_failure_preserves_applying_recovery(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")

    def fail_destination_cleanup(**_kwargs):
        raise RuntimeError("destination cleanup interrupted")

    state.clear_pending_personal_context_outbox = fail_destination_cleanup
    custodian = InMemoryPersonalContextLinkKeyCustodian()
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        first_link_sync=FakeFirstLinkSync(),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()

    with pytest.raises(RuntimeError, match="destination cleanup interrupted"):
        await service.apply(plan.plan_id, {})

    applying = state.get_personal_context_link_state(**SCOPE)
    assert applying["state"] == "applying"
    assert profile.frozen_plan_id == plan.plan_id
    assert custodian.load(**service._key_binding(applying)) == b"s" * 32


@pytest.mark.asyncio
async def test_precommit_interruption_discards_only_the_uncommitted_staged_key(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    custodian = InMemoryPersonalContextLinkKeyCustodian()
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    plan = await service.plan()
    review = state.get_personal_context_link_state(**SCOPE)
    applying = state.set_personal_context_link_state(
        **SCOPE,
        **{
            key: review[key]
            for key in (
                "device_id",
                "dataset_id",
                "authority_id",
                "profile_id",
                "integrity_key_id",
                "key_record_id",
                "purge_generation",
                "bootstrap_cursor",
                "plan_id",
                "rebaseline_version",
                "attention_code",
            )
        },
        state="applying",
        expected_states=("review_required",),
    )
    binding = service._key_binding(applying)
    custodian.stage(**binding, integrity_key=b"s" * 32)

    assert service.abandon_uncommitted_apply() is True
    assert state.get_personal_context_link_state(**SCOPE)["state"] == (
        "attention_required"
    )
    with pytest.raises(ValueError, match="binding_mismatch"):
        custodian.load(**binding)

    retry = await service.plan()
    assert retry.plan_id != plan.plan_id


@pytest.mark.asyncio
async def test_restart_preserves_recovery_material_when_profile_cannot_compose(
    tmp_path,
) -> None:
    profile = FakeProfileService()
    state = SyncStateRepository(tmp_path / "sync.db")
    custodian = InMemoryPersonalContextLinkKeyCustodian()
    fallback_releases: list[str] = []
    service = PersonalContextLinkService(
        personal_context_service=profile,
        server_sync_service=FakeServerSync(),
        state_repository=state,
        wrapping_key_provider=FakeWrappingProvider(),
        key_custodian=custodian,
        freeze_release_fallback=lambda plan_id: fallback_releases.append(plan_id),
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        display_name="Laptop",
    )
    await service.plan()
    review = state.get_personal_context_link_state(**SCOPE)
    applying = state.set_personal_context_link_state(
        **SCOPE,
        **{
            key: review[key]
            for key in (
                "device_id",
                "dataset_id",
                "authority_id",
                "profile_id",
                "integrity_key_id",
                "key_record_id",
                "purge_generation",
                "bootstrap_cursor",
                "plan_id",
                "rebaseline_version",
                "attention_code",
            )
        },
        state="applying",
        expected_states=("review_required",),
    )
    binding = service._key_binding(applying)
    custodian.stage(**binding, integrity_key=b"s" * 32)
    service._profile = PersonalContextService.locked("profile_key_unavailable")

    assert service.abandon_uncommitted_apply() is False
    interrupted = state.get_personal_context_link_state(**SCOPE)
    assert interrupted["state"] == "applying"
    assert fallback_releases == []
    assert custodian.load(**binding) == b"s" * 32
