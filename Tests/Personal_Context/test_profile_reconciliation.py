from __future__ import annotations

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
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from tldw_chatbook.Personal_Context.reconciliation import (
    CanonicalBootstrapSnapshot,
    build_reconciliation_plan,
)
from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
    ProfileLockedError,
)
from tldw_chatbook.Personal_Context.repository import (
    PersonalContextRepository,
    ProfileKeyActivationPendingError,
)


NOW = "2026-08-30T12:00:00.000Z"


def _manifest(profile_id: str, version: str) -> ProfileManifest:
    return ProfileManifest(
        profile_id=profile_id,
        revision=1,
        purge_generation=0,
        created_at=NOW,
        updated_at=NOW,
        current_version_id=version,
    )


def _scope(profile_id: str, scope_id: str, kind: ScopeKind) -> ProfileScope:
    return ProfileScope(
        profile_id=profile_id,
        scope_id=scope_id,
        kind=kind,
        version_id=f"{scope_id}-version",
        created_at=NOW,
        updated_at=NOW,
    )


def _record(
    profile_id: str,
    scope_id: str,
    record_id: str,
    *,
    subject: str,
    value: str,
    version: str,
    sync_mode: SyncMode = SyncMode.SYNCABLE,
) -> ProfileRecord:
    return ProfileRecord(
        profile_id=profile_id,
        record_id=record_id,
        scope_id=scope_id,
        kind="preference",
        payload=PreferencePayload(subject=subject, polarity="like", value=value),
        semantic_key=SemanticKey(namespace="preference", subject=subject),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=sync_mode,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
        provenance=ProfileProvenance(
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id=version,
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _snapshot(
    *,
    scopes: tuple[ProfileScope, ...],
    records: tuple[ProfileRecord, ...],
    proposals: tuple[ProfileProposal, ...] = (),
) -> CanonicalBootstrapSnapshot:
    return CanonicalBootstrapSnapshot(
        dataset_id="dataset-1",
        authority_id="authority-1",
        manifest=_manifest("profile-server", "manifest-server"),
        scopes=scopes,
        records=records,
        proposals=proposals,
        purge_generation=0,
        schema_version=1,
        quotas={"max_record_bytes": 16_384},
        cursor="sha256:" + "a" * 64,
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        wrapped_key_blob="wrapped-private-material",
    )


def test_plan_is_content_free_read_only_and_requires_exact_collision_review() -> None:
    local_global = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    remote_global = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    local = _record(
        "profile-local",
        local_global.scope_id,
        "record-local",
        subject="response.detail",
        value="concise",
        version="record-local-v1",
    )
    remote = _record(
        "profile-server",
        remote_global.scope_id,
        "record-server",
        subject="response.detail",
        value="detailed",
        version="record-server-v1",
    )

    plan = build_reconciliation_plan(
        local_manifest=_manifest("profile-local", "manifest-local"),
        local_scopes=(local_global,),
        local_records=(local,),
        local_proposals=(),
        remote=_snapshot(scopes=(remote_global,), records=(remote,)),
        local_workspace_bindings={},
    )

    assert plan.global_scope_mapping == (
        "scope-local-global",
        "scope-server-global",
    )
    assert plan.key_collisions[0].record_ids == (
        "record-local",
        "record-server",
    )
    assert plan.required_decision_ids == (plan.key_collisions[0].decision_id,)
    rendered = repr(plan)
    assert "concise" not in rendered
    assert "detailed" not in rendered
    assert "wrapped-private-material" not in rendered


def test_plan_excludes_device_only_and_keeps_remote_workspaces_unlinked() -> None:
    local_global = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    remote_global = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    remote_workspace = _scope(
        "profile-server", "scope-server-workspace", ScopeKind.WORKSPACE
    )
    private = _record(
        "profile-local",
        local_global.scope_id,
        "record-private",
        subject="secret.preference",
        value="local only",
        version="record-private-v1",
        sync_mode=SyncMode.DEVICE_ONLY,
    )
    remote = _record(
        "profile-server",
        remote_workspace.scope_id,
        "record-remote-workspace",
        subject="project.detail",
        value="context",
        version="record-remote-v1",
    )

    plan = build_reconciliation_plan(
        local_manifest=_manifest("profile-local", "manifest-local"),
        local_scopes=(local_global,),
        local_records=(private,),
        local_proposals=(),
        remote=_snapshot(
            scopes=(remote_global, remote_workspace), records=(remote,)
        ),
        local_workspace_bindings={},
    )

    assert plan.device_only_record_ids == ("record-private",)
    assert plan.local_only_record_ids == ()
    assert plan.unlinked_remote_scope_ids == ("scope-server-workspace",)
    assert plan.remote_only_record_ids == ("record-remote-workspace",)


def test_reviewed_apply_adopts_server_identity_and_rebaselines_every_artifact(
    tmp_path,
    proposal_factory,
) -> None:
    protector = InMemoryProfileKeyProtector()
    repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=protector
    )
    local_manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    local_record = _record(
        "profile-local",
        local_scope.scope_id,
        "record-local",
        subject="response.detail",
        value="concise",
        version="record-local-v1",
    )
    repository.create_profile_with_global_scope(local_manifest, local_scope)
    repository.commit_record_version(
        local_record,
        expected_version_id=None,
        outbox_body={"version": 1, "record": local_record.model_dump(mode="json")},
    )
    local_proposal = proposal_factory(
        local_manifest.profile_id, proposal_id="proposal-local"
    )
    assert local_proposal.proposed_record is not None
    local_proposal = local_proposal.model_copy(
        update={
            "scope_id": local_scope.scope_id,
            "proposed_record": local_proposal.proposed_record.model_copy(
                update={"scope_id": local_scope.scope_id}
            ),
        }
    )
    repository.commit_proposal(local_proposal)
    remote_scope = _scope(
        "profile-server", "scope-server-global", ScopeKind.GLOBAL
    )
    remote_proposal = proposal_factory(
        "profile-server", proposal_id="proposal-server"
    )
    assert remote_proposal.proposed_record is not None
    remote_proposal = remote_proposal.model_copy(
        update={
            "scope_id": remote_scope.scope_id,
            "proposed_record": remote_proposal.proposed_record.model_copy(
                update={
                    "profile_id": "profile-server",
                    "scope_id": remote_scope.scope_id,
                }
            ),
        }
    )
    remote = _snapshot(
        scopes=(remote_scope,), records=(), proposals=(remote_proposal,)
    )
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=(local_scope,),
        local_records=(local_record,),
        local_proposals=(local_proposal,),
        remote=remote,
        local_workspace_bindings={},
    )
    old_material = repository._keys
    assert old_material is not None

    result = repository.apply_reviewed_link(
        plan=plan,
        remote=remote,
        decisions={},
        integrity_key=b"s" * 32,
    )

    assert result == {"rebaseline_version": old_material.key_version + 1}
    assert repository.get_manifest() == remote.manifest
    retained = repository.get_record("record-local")
    assert retained is not None
    assert retained.profile_id == "profile-server"
    assert retained.scope_id == "scope-server-global"
    assert retained.version_id == "record-local-v1"
    retained_proposals = {
        proposal.proposal_id: proposal for proposal in repository.list_proposals()
    }
    assert set(retained_proposals) == {"proposal-local", "proposal-server"}
    assert all(
        proposal.profile_id == "profile-server"
        and proposal.scope_id == "scope-server-global"
        for proposal in retained_proposals.values()
    )
    assert repository._keys is not None
    assert repository._keys.encryption_key == old_material.encryption_key
    assert repository._keys.integrity_key == b"s" * 32
    with repository._connect() as connection:
        versions = {
            row[0]
            for row in connection.execute("SELECT key_version FROM encrypted_objects")
        }
        retained_head_types = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT object_type FROM object_heads"
            )
        }
    assert versions == {old_material.key_version + 1}
    assert {"manifest", "scope", "record", "proposal", "outbox"}.issubset(
        retained_head_types
    )
    assert PersonalContextRepository(
        tmp_path / "profile.db", key_protector=protector
    ).get_manifest() == remote.manifest


def test_reviewed_apply_rejects_a_local_edit_racing_with_review(tmp_path) -> None:
    protector = InMemoryProfileKeyProtector()
    repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=protector
    )
    local_manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(local_manifest, local_scope)
    remote_scope = _scope(
        "profile-server", "scope-server-global", ScopeKind.GLOBAL
    )
    remote = _snapshot(scopes=(remote_scope,), records=())
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=(local_scope,),
        local_records=(),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )
    racing_record = _record(
        "profile-local",
        local_scope.scope_id,
        "record-racing",
        subject="new.detail",
        value="new",
        version="record-racing-v1",
    )
    repository.commit_record_version(
        racing_record,
        expected_version_id=None,
        outbox_body={"version": 1, "record": racing_record.model_dump(mode="json")},
    )

    with pytest.raises(ValueError, match="link_plan_stale"):
        repository.apply_reviewed_link(
            plan=plan,
            remote=remote,
            decisions={},
            integrity_key=b"s" * 32,
        )

    assert repository.get_manifest() == local_manifest
    assert repository.get_record("record-racing") == racing_record


class _FailOnceActivationProtector(InMemoryProfileKeyProtector):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_activation = True

    def replace(self, profile_ref, material) -> None:
        if self.fail_next_activation:
            self.fail_next_activation = False
            raise ProfileLockedError("injected secure-custody interruption")
        super().replace(profile_ref, material)


def test_staged_integrity_key_recovers_interruption_after_database_rebaseline(
    tmp_path,
) -> None:
    protector = _FailOnceActivationProtector()
    db_path = tmp_path / "profile.db"
    repository = PersonalContextRepository(db_path, key_protector=protector)
    local_manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(local_manifest, local_scope)
    remote_scope = _scope(
        "profile-server", "scope-server-global", ScopeKind.GLOBAL
    )
    remote = _snapshot(scopes=(remote_scope,), records=())
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=(local_scope,),
        local_records=(),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )

    with pytest.raises(ProfileKeyActivationPendingError):
        repository.apply_reviewed_link(
            plan=plan,
            remote=remote,
            decisions={},
            integrity_key=b"s" * 32,
        )

    with pytest.raises(ProfileLockedError):
        PersonalContextRepository(
            db_path, key_protector=protector
        ).get_manifest()

    recovered = PersonalContextRepository(
        db_path,
        key_protector=protector,
        recovery_integrity_key=b"s" * 32,
        expected_recovery_profile_id="profile-server",
    )

    assert recovered.get_manifest() == remote.manifest
    assert PersonalContextRepository(
        db_path, key_protector=protector
    ).get_manifest() == remote.manifest
