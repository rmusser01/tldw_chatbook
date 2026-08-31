from __future__ import annotations

import pytest
from tldw_profile_core import ProfileManifest, ProfileScope, ScopeKind

from tldw_chatbook.Personal_Context.link_key_custody import (
    InMemoryPersonalContextLinkKeyCustodian,
)
from tldw_chatbook.Personal_Context.link_service import PersonalContextLinkService
from tldw_chatbook.Personal_Context.repository import (
    ProfileKeyActivationPendingError,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


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
        plan_id="plan-1",
        rebaseline_version=1,
        attention_code=None,
    )
    assert reopened.personal_context_sync_enabled(**SCOPE) is True


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


class FakeServerSync:
    def __init__(self) -> None:
        self.bootstrap_calls = []
        self.complete_calls = []
        self.fail_complete_once = False

    async def bootstrap_personal_context_link(self, **kwargs):
        self.bootstrap_calls.append(kwargs)
        manifest = _manifest("profile-server", "manifest-server")
        scope = _scope("profile-server", "scope-server")
        return {
            "device_id": "device-1",
            "dataset_id": "dataset-1",
            "authority_id": "authority-1",
            "manifest": manifest.model_dump(mode="json"),
            "scopes": [scope.model_dump(mode="json")],
            "records": [],
            "proposals": [],
            "purge_generation": 0,
            "schema_version": 1,
            "quotas": {"max_record_bytes": 16_384},
            "cursor": "sha256:" + "e" * 64,
            "integrity_key_id": "integrity-1",
            "key_record_id": "key-record-1",
            "wrapped_key_blob": "wrapped",
        }

    async def complete_personal_context_link(self, **kwargs):
        self.complete_calls.append(kwargs)
        if self.fail_complete_once:
            self.fail_complete_once = False
            raise RuntimeError("temporary failure")


class FakeWrappingProvider:
    public_key_pem = "public"

    def unwrap_integrity_key(self, blob, *, integrity_key_id):
        assert blob == "wrapped"
        assert integrity_key_id == "integrity-1"
        return b"s" * 32


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
    assert state.personal_context_sync_enabled(**SCOPE) is True


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
    assert state.personal_context_sync_enabled(**SCOPE) is True
    with pytest.raises(ValueError, match="binding_mismatch"):
        custodian.load(**service._key_binding(interrupted))


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
