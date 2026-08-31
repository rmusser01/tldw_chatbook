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
    PersonalContextLinkInProgressError,
    ProfileKeyActivationPendingError,
)
from tldw_chatbook.Personal_Context.sync_outbox import ProfileSyncOutbox
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    RecordMutation,
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
    schema_version: int = 1,
    quotas: dict[str, int] | None = None,
    purge_generation: int = 0,
) -> CanonicalBootstrapSnapshot:
    return CanonicalBootstrapSnapshot(
        dataset_id="dataset-1",
        authority_id="authority-1",
        manifest=_manifest("profile-server", "manifest-server"),
        scopes=scopes,
        records=records,
        proposals=proposals,
        purge_generation=purge_generation,
        schema_version=schema_version,
        quotas=quotas or {"max_record_bytes": 16_384},
        cursor="sha256:" + "a" * 64,
        integrity_key_id="key-1",
        key_record_id="key-record-1",
        wrapped_key_blob="wrapped-private-material",
    )


def _freeze(repository: PersonalContextRepository, plan) -> None:
    repository.acquire_first_link_freeze(
        plan_id=plan.plan_id,
        snapshot_token=plan.local_snapshot_token,
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


def test_plan_records_exact_content_free_contract_outcomes() -> None:
    local_global = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    remote_global = _scope(
        "profile-server", "scope-server-global", ScopeKind.GLOBAL
    )

    plan = build_reconciliation_plan(
        local_manifest=_manifest("profile-local", "manifest-local"),
        local_scopes=(local_global,),
        local_records=(),
        local_proposals=(),
        remote=_snapshot(
            scopes=(remote_global,),
            records=(),
            schema_version=3,
            quotas={"max_record_bytes": 32_768, "max_search_results": 40},
        ),
        local_workspace_bindings={},
        required_schema_version=3,
        required_quotas={"max_record_bytes": 16_384, "max_search_results": 20},
    )

    assert plan.profile_adoption == ("profile-local", "profile-server")
    assert plan.global_scope_mapping == (
        "scope-local-global",
        "scope-server-global",
    )
    assert plan.schema_outcome == "compatible:3"
    assert plan.quota_outcome == "minimums_satisfied"
    assert plan.required_quotas == {
        "max_record_bytes": 16_384,
        "max_search_results": 20,
    }
    assert plan.server_quotas == {
        "max_record_bytes": 32_768,
        "max_search_results": 40,
    }
    assert plan.purge_outcome == "generation_matches:0"


def test_review_freeze_blocks_mutations_survives_restart_and_cancel_releases(
    tmp_path,
) -> None:
    protector = InMemoryProfileKeyProtector()
    db_path = tmp_path / "profile.db"
    repository = PersonalContextRepository(db_path, key_protector=protector)
    local_manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(local_manifest, local_scope)
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=(local_scope,),
        local_records=(),
        local_proposals=(),
        remote=_snapshot(
            scopes=(
                _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL),
            ),
            records=(),
        ),
        local_workspace_bindings={},
    )
    repository.acquire_first_link_freeze(
        plan_id=plan.plan_id,
        snapshot_token=plan.local_snapshot_token,
    )
    blocked = _record(
        "profile-local",
        "scope-local-global",
        "record-blocked",
        subject="response.detail",
        value="concise",
        version="record-blocked-v1",
    )

    assert repository.get_manifest() == local_manifest
    with pytest.raises(
        PersonalContextLinkInProgressError,
        match="personal_context_link_in_progress",
    ):
        repository.commit_record_version(blocked, expected_version_id=None)

    reopened = PersonalContextRepository(db_path, key_protector=protector)
    with pytest.raises(
        PersonalContextLinkInProgressError,
        match="personal_context_link_in_progress",
    ):
        reopened.commit_record_version(blocked, expected_version_id=None)
    assert reopened.release_first_link_freeze(plan_id="wrong-plan") is False
    assert reopened.release_first_link_freeze(plan_id=plan.plan_id) is True
    reopened.commit_record_version(blocked, expected_version_id=None)


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


def test_plan_blocks_remote_record_that_reuses_device_only_identity() -> None:
    local_global = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    remote_global = _scope(
        "profile-server", "scope-server-global", ScopeKind.GLOBAL
    )
    private = _record(
        "profile-local",
        local_global.scope_id,
        "record-shared-private",
        subject="secret.preference",
        value="never replace",
        version="private-v1",
        sync_mode=SyncMode.DEVICE_ONLY,
    )
    remote = _record(
        "profile-server",
        remote_global.scope_id,
        private.record_id,
        subject="public.preference",
        value="server",
        version="server-v1",
    )

    plan = build_reconciliation_plan(
        local_manifest=_manifest("profile-local", "manifest-local"),
        local_scopes=(local_global,),
        local_records=(private,),
        local_proposals=(),
        remote=_snapshot(scopes=(remote_global,), records=(remote,)),
        local_workspace_bindings={},
    )

    assert "device_only_identity_collision" in plan.attention_codes
    assert plan.remote_only_record_ids == ()
    assert (
        private.record_id,
        "device_only_identity_attention",
        private.version_id,
        remote.version_id,
    ) in plan.record_outcomes
    assert plan.can_approve is False


def test_plan_preallocates_new_workspace_identity_and_mapping_collisions() -> None:
    local_global = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    local_workspace = _scope(
        "profile-local", "scope-local-workspace", ScopeKind.WORKSPACE
    )
    remote_global = _scope(
        "profile-server", "scope-server-global", ScopeKind.GLOBAL
    )
    remote_workspace = _scope(
        "profile-server", "scope-server-workspace", ScopeKind.WORKSPACE
    )
    local = _record(
        "profile-local",
        local_workspace.scope_id,
        "record-local-workspace",
        subject="project.goal",
        value="local",
        version="local-v1",
    )
    remote = _record(
        "profile-server",
        remote_workspace.scope_id,
        "record-remote-workspace",
        subject="project.goal",
        value="remote",
        version="remote-v1",
    )
    kwargs = {
        "local_manifest": _manifest("profile-local", "manifest-local"),
        "local_scopes": (local_global, local_workspace),
        "local_records": (local,),
        "local_proposals": (),
        "remote": _snapshot(
            scopes=(remote_global, remote_workspace), records=(remote,)
        ),
        "local_workspace_bindings": {},
        "plan_id": "pc-link-plan-reviewed",
    }

    plan = build_reconciliation_plan(**kwargs)
    replay = build_reconciliation_plan(**kwargs)

    new_scope_id = dict(plan.workspace_new_scope_ids)[local_workspace.scope_id]
    assert new_scope_id.startswith("scope-workspace-")
    assert new_scope_id != local_workspace.scope_id
    assert replay.workspace_new_scope_ids == plan.workspace_new_scope_ids
    conflict = plan.workspace_mapping_conflicts[0]
    assert conflict.local_scope_id == local_workspace.scope_id
    assert conflict.remote_scope_id == remote_workspace.scope_id
    assert conflict.record_ids == (
        "record-local-workspace",
        "record-remote-workspace",
    )


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

    _freeze(repository, plan)
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
    outbox = ProfileSyncOutbox(repository)
    pending = outbox.list_pending()
    assert {(entry.object_type, entry.object_id, entry.version_id) for entry in pending} == {
        ("record", "record-local", "record-local-v1"),
        (
            "proposal",
            "proposal-local",
            next(
                version
                for object_type, object_id, version in repository.first_link_head_rows()
                if object_type == "proposal" and object_id == "proposal-local"
            ),
        ),
    }
    for entry in pending:
        body = outbox.read_body(entry.outbox_id)
        assert body is not None
        assert body[entry.object_type]["profile_id"] == "profile-server"
        assert body[entry.object_type]["scope_id"] == "scope-server-global"
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
    _freeze(repository, plan)
    with pytest.raises(
        PersonalContextLinkInProgressError,
        match="personal_context_link_in_progress",
    ):
        repository.commit_record_version(
            racing_record,
            expected_version_id=None,
            outbox_body={
                "version": 1,
                "record": racing_record.model_dump(mode="json"),
            },
        )

    assert repository.get_manifest() == local_manifest
    assert repository.get_record("record-racing") is None


def test_reviewed_apply_never_rewrites_user_strings_equal_to_old_id(tmp_path) -> None:
    transformed = PersonalContextRepository._transform_link_body(
        "record",
        {
            "profile_id": "profile-local",
            "scope_id": "scope-local-global",
            "payload": {"value": "profile-local"},
            "provenance": {"reason_code": "scope-local-global"},
        },
        old_profile_id="profile-local",
        new_profile_id="profile-server",
        scope_mapping={"scope-local-global": "scope-server-global"},
    )
    assert transformed["payload"]["value"] == "profile-local"
    assert transformed["provenance"]["reason_code"] == "scope-local-global"

    repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=InMemoryProfileKeyProtector()
    )
    local_manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    record = _record(
        local_manifest.profile_id,
        local_scope.scope_id,
        "record-canary",
        subject="identity.canary",
        value=local_manifest.profile_id,
        version="record-canary-v1",
    )
    record = record.model_copy(
        update={
            "provenance": record.provenance.model_copy(
                update={"reason_code": local_scope.scope_id}
            )
        }
    )
    repository.create_profile_with_global_scope(local_manifest, local_scope)
    repository.commit_record_version(
        record,
        expected_version_id=None,
        outbox_body={"version": 1, "record": record.model_dump(mode="json")},
    )
    remote_scope = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    remote = _snapshot(scopes=(remote_scope,), records=())
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=(local_scope,),
        local_records=(record,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )

    _freeze(repository, plan)
    repository.apply_reviewed_link(
        plan=plan, remote=remote, decisions={}, integrity_key=b"s" * 32
    )

    retained = repository.get_record(record.record_id)
    assert retained is not None
    assert retained.profile_id == "profile-server"
    assert retained.scope_id == "scope-server-global"
    assert retained.payload.value == "profile-local"
    assert retained.provenance.reason_code == "scope-local-global"


@pytest.mark.parametrize("decision", ["unlinked", "new"])
def test_workspace_link_decision_is_explicit_and_never_reuses_provisional_id(
    tmp_path, decision
) -> None:
    repository = PersonalContextRepository(
        tmp_path / f"{decision}.db", key_protector=InMemoryProfileKeyProtector()
    )
    manifest = _manifest("profile-local", "manifest-local")
    global_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    workspace = _scope("profile-local", "scope-provisional", ScopeKind.WORKSPACE)
    repository.create_profile_with_global_scope(manifest, global_scope)
    repository.commit_scope_with_binding(
        workspace,
        {"version": 1, "local_workspace_id": "workspace-local", "label": "Project"},
    )
    record = _record(
        manifest.profile_id,
        workspace.scope_id,
        "workspace-record",
        subject="project.goal",
        value="ship",
        version="workspace-record-v1",
    )
    repository.commit_record_version(
        record,
        expected_version_id=None,
        outbox_body={"version": 1, "record": record.model_dump(mode="json")},
    )
    remote_global = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    remote = _snapshot(scopes=(remote_global,), records=())
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(global_scope, workspace),
        local_records=(record,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings=repository.list_validated_scope_bindings(),
    )

    _freeze(repository, plan)
    repository.apply_reviewed_link(
        plan=plan,
        remote=remote,
        decisions={f"workspace:{workspace.scope_id}": decision},
        integrity_key=b"s" * 32,
    )

    retained = repository.get_record(record.record_id)
    assert retained is not None
    if decision == "unlinked":
        assert retained.scope_id == workspace.scope_id
        assert workspace.scope_id not in repository.list_validated_scope_bindings()
        assert record.record_id not in repository.first_link_sync_heads()[
            "personal_context.record"
        ]
    else:
        assert retained.scope_id == dict(plan.workspace_new_scope_ids)[
            workspace.scope_id
        ]
        assert retained.scope_id in repository.list_validated_scope_bindings()


def test_workspace_link_decision_cannot_target_remote_global_scope(tmp_path) -> None:
    repository = PersonalContextRepository(
        tmp_path / "global-target.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    manifest = _manifest("profile-local", "manifest-local")
    global_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    workspace = _scope("profile-local", "scope-local-workspace", ScopeKind.WORKSPACE)
    repository.create_profile_with_global_scope(manifest, global_scope)
    repository.commit_scope_with_binding(
        workspace,
        {"version": 1, "local_workspace_id": "workspace-local", "label": "Project"},
    )
    remote_global = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    remote = _snapshot(scopes=(remote_global,), records=())
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(global_scope, workspace),
        local_records=(),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings=repository.list_validated_scope_bindings(),
    )
    _freeze(repository, plan)

    with pytest.raises(ValueError, match="workspace_mapping_invalid"):
        repository.apply_reviewed_link(
            plan=plan,
            remote=remote,
            decisions={f"workspace:{workspace.scope_id}": remote_global.scope_id},
            integrity_key=b"s" * 32,
        )

    assert repository.get_manifest() == manifest
    assert repository.get_scope(workspace.scope_id) == workspace


def test_replan_preserves_an_already_adopted_bound_workspace_identity(tmp_path) -> None:
    repository = PersonalContextRepository(
        tmp_path / "workspace-replan.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    manifest = _manifest("profile-local", "manifest-local")
    global_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    workspace = _scope("profile-local", "scope-provisional", ScopeKind.WORKSPACE)
    repository.create_profile_with_global_scope(manifest, global_scope)
    repository.commit_scope_with_binding(
        workspace,
        {"version": 1, "local_workspace_id": "workspace-local", "label": "Project"},
    )
    remote_global = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    first_remote = _snapshot(scopes=(remote_global,), records=())
    first_plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(global_scope, workspace),
        local_records=(),
        local_proposals=(),
        remote=first_remote,
        local_workspace_bindings=repository.list_validated_scope_bindings(),
    )
    adopted_scope_id = dict(first_plan.workspace_new_scope_ids)[workspace.scope_id]
    _freeze(repository, first_plan)
    repository.apply_reviewed_link(
        plan=first_plan,
        remote=first_remote,
        decisions={f"workspace:{workspace.scope_id}": "new"},
        integrity_key=b"s" * 32,
    )
    adopted_scope = repository.get_scope(adopted_scope_id)
    assert adopted_scope is not None
    retry_remote = _snapshot(
        scopes=(remote_global, adopted_scope),
        records=(),
    )

    retry_plan = build_reconciliation_plan(
        local_manifest=repository.get_manifest(),
        local_scopes=tuple(repository.list_scopes()),
        local_records=tuple(repository.list_records()),
        local_proposals=tuple(repository.list_proposals()),
        remote=retry_remote,
        local_workspace_bindings=repository.list_validated_scope_bindings(),
    )

    assert retry_plan.local_workspace_scope_ids == ()
    assert retry_plan.workspace_new_scope_ids == ()
    assert retry_plan.required_decision_ids == ()


def test_reviewed_link_rebinds_retained_undo_identity_and_keeps_it_usable(
    tmp_path,
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "undo-rebind.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local-global", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(manifest, local_scope)
    first = _record(
        manifest.profile_id,
        local_scope.scope_id,
        "record-local",
        subject="response.detail",
        value="before-link",
        version="record-v1",
    )
    repository.commit_record_version(first, expected_version_id=None)
    service = PersonalContextService(repository)
    changed = service.update_record(
        first.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail",
                polarity="like",
                value="after-local-edit",
            )
        ),
        expected_version_id=first.version_id,
    )
    undo_id = service.list_undo_ids()[0]
    remote_scope = _scope("profile-server", "scope-server-global", ScopeKind.GLOBAL)
    remote = _snapshot(scopes=(remote_scope,), records=())
    plan = build_reconciliation_plan(
        local_manifest=repository.get_manifest(),
        local_scopes=tuple(repository.list_scopes()),
        local_records=(changed,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )
    _freeze(repository, plan)
    repository.apply_reviewed_link(
        plan=plan,
        remote=remote,
        decisions={},
        integrity_key=b"s" * 32,
    )
    assert repository.release_first_link_freeze(plan_id=plan.plan_id) is True

    restored = service.undo(undo_id)

    assert restored.profile_id == "profile-server"
    assert restored.scope_id == "scope-server-global"
    assert restored.payload.value == "before-link"


def test_local_only_multiversion_record_journals_oldest_to_head(tmp_path) -> None:
    repository = PersonalContextRepository(
        tmp_path / "lineage.db", key_protector=InMemoryProfileKeyProtector()
    )
    manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(manifest, local_scope)
    first = _record(
        manifest.profile_id,
        local_scope.scope_id,
        "record-local",
        subject="response.detail",
        value="concise",
        version="record-v1",
    )
    second = first.model_copy(
        update={
            "version_id": "record-v2",
            "parent_version_id": "record-v1",
            "payload": PreferencePayload(
                subject="response.detail", polarity="like", value="very concise"
            ),
        }
    )
    repository.commit_record_version(
        first,
        expected_version_id=None,
        outbox_body={"version": 1, "record": first.model_dump(mode="json")},
    )
    repository.commit_record_version(
        second,
        expected_version_id="record-v1",
        outbox_body={"version": 1, "record": second.model_dump(mode="json")},
    )
    remote_scope = _scope("profile-server", "scope-server", ScopeKind.GLOBAL)
    remote = _snapshot(scopes=(remote_scope,), records=())
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(local_scope,),
        local_records=(second,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )

    _freeze(repository, plan)
    repository.apply_reviewed_link(
        plan=plan, remote=remote, decisions={}, integrity_key=b"s" * 32
    )

    outbox = ProfileSyncOutbox(repository)
    versions = [
        entry.version_id
        for entry in outbox.list_pending()
        if entry.object_type == "record" and entry.object_id == "record-local"
    ]
    assert versions == ["record-v1", "record-v2"]


def test_local_same_id_winner_is_rebased_on_server_head(tmp_path) -> None:
    repository = PersonalContextRepository(
        tmp_path / "merge.db", key_protector=InMemoryProfileKeyProtector()
    )
    manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(manifest, local_scope)
    local = _record(
        manifest.profile_id,
        local_scope.scope_id,
        "record-shared",
        subject="response.detail",
        value="local",
        version="local-v1",
    )
    repository.commit_record_version(
        local,
        expected_version_id=None,
        outbox_body={"version": 1, "record": local.model_dump(mode="json")},
    )
    remote_scope = _scope("profile-server", "scope-server", ScopeKind.GLOBAL)
    remote_record = _record(
        "profile-server",
        remote_scope.scope_id,
        "record-shared",
        subject="response.detail",
        value="server",
        version="server-v3",
    )
    remote = _snapshot(scopes=(remote_scope,), records=(remote_record,))
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(local_scope,),
        local_records=(local,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )

    _freeze(repository, plan)
    repository.apply_reviewed_link(
        plan=plan,
        remote=remote,
        decisions={plan.version_conflicts[0].decision_id: "local"},
        integrity_key=b"s" * 32,
    )

    merged = repository.get_record("record-shared")
    assert merged is not None
    assert merged.payload.value == "local"
    assert merged.parent_version_id == "server-v3"
    assert merged.version_id not in {"local-v1", "server-v3"}
    pending = [
        entry
        for entry in ProfileSyncOutbox(repository).list_pending()
        if entry.object_id == "record-shared"
    ]
    assert [entry.version_id for entry in pending] == [merged.version_id]


def test_local_collision_winner_journals_tombstone_for_remote_occupant(tmp_path) -> None:
    repository = PersonalContextRepository(
        tmp_path / "tombstone.db", key_protector=InMemoryProfileKeyProtector()
    )
    manifest = _manifest("profile-local", "manifest-local")
    local_scope = _scope("profile-local", "scope-local", ScopeKind.GLOBAL)
    repository.create_profile_with_global_scope(manifest, local_scope)
    local = _record(
        manifest.profile_id,
        local_scope.scope_id,
        "record-local",
        subject="response.detail",
        value="local",
        version="local-v1",
    )
    repository.commit_record_version(
        local,
        expected_version_id=None,
        outbox_body={"version": 1, "record": local.model_dump(mode="json")},
    )
    remote_scope = _scope("profile-server", "scope-server", ScopeKind.GLOBAL)
    remote_record = _record(
        "profile-server",
        remote_scope.scope_id,
        "record-remote",
        subject="response.detail",
        value="server",
        version="remote-v4",
    )
    remote = _snapshot(scopes=(remote_scope,), records=(remote_record,))
    plan = build_reconciliation_plan(
        local_manifest=manifest,
        local_scopes=(local_scope,),
        local_records=(local,),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )

    _freeze(repository, plan)
    repository.apply_reviewed_link(
        plan=plan,
        remote=remote,
        decisions={plan.key_collisions[0].decision_id: "local"},
        integrity_key=b"s" * 32,
    )

    tombstone = repository.get_record("record-remote")
    assert tombstone is not None
    assert tombstone.state is RecordState.DELETED
    assert tombstone.parent_version_id == "remote-v4"
    assert tombstone.payload is None
    pending_ids = {
        entry.object_id for entry in ProfileSyncOutbox(repository).list_pending()
    }
    assert {"record-local", "record-remote"}.issubset(pending_ids)


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

    _freeze(repository, plan)
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
