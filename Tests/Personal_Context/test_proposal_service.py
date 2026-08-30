from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tldw_profile_core import (
    ActorType,
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfilePromoteRequest,
    ProfileProposeRequest,
    ProfileUpdateRequest,
    ProposalState,
    SyncMode,
    WorkingContextPayload,
)

from tldw_chatbook.Personal_Context.proposal_service import (
    ProfileProposalQuota,
    ProfileProposalService,
)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import (
    AgentAuthority,
    PersonalContextAuthorityError,
)
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_chatbook.Personal_Context.service import ProfileConflictError, RecordMutation


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _harness(tmp_path, memory_protector, *, clock=lambda: NOW):
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(
        repository,
        clock=clock,
        id_factory=Ids(),
    )
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)
    return (
        repository,
        service,
        service.proposal_service(quota=ProfileProposalQuota(per_session=300)),
        manifest,
        scope,
    )


def _create_request(value: str = "concise") -> ProfileProposeRequest:
    return ProfileProposeRequest(
        operation="create",
        proposed_payload=PreferencePayload(
            subject="response.detail", polarity="like", value=value
        ),
    )


def test_service_owned_proposal_collaborator_creates_pending_canonical_proposal(
    tmp_path, memory_protector
) -> None:
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )

    proposal = proposals.create(
        _create_request(),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )

    assert isinstance(proposals, ProfileProposalService)
    assert proposal.state.value == "pending"
    assert proposal.proposed_record is not None
    assert proposal.proposed_record.controls.model_dump(mode="json") == {
        "sync_mode": "syncable",
        "agent_visibility": "agent_visible",
    }
    assert proposal.proposed_record.semantic_key.model_dump(mode="json") == {
        "namespace": "preference",
        "subject": "response.detail",
    }
    assert service.list_records(scope_ids=(scope.scope_id,)) == ()
    assert proposals.list_pending() == (proposal,)


@pytest.mark.parametrize(
    ("action", "state"),
    [
        ("reject", ProposalState.REJECTED),
        ("supersede", ProposalState.SUPERSEDED),
        ("expire", ProposalState.EXPIRED),
    ],
)
def test_terminal_resolution_shreds_all_prior_content_bearing_ciphertext(
    tmp_path, memory_protector, action: str, state: ProposalState
) -> None:
    repository, _service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    proposal = proposals.create(
        _create_request("PROPOSAL-CANARY-7b77"),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )

    receipt = getattr(proposals, action)(proposal.proposal_id)

    assert receipt.state is state
    assert receipt.proposed_record is None
    assert receipt.confidence is None
    with sqlite3.connect(repository.db_path) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM encrypted_objects "
            "WHERE object_type = 'proposal' AND object_id = ?",
            (proposal.proposal_id,),
        ).fetchone()[0]
    assert rows == 1


def test_repository_enforces_unresolved_ceiling_inside_commit_transaction(
    tmp_path, memory_protector
) -> None:
    from tldw_chatbook.Personal_Context.repository import (
        MAX_UNRESOLVED_PROPOSALS,
        ProposalLimitExceededError,
    )

    repository, _service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    assert MAX_UNRESOLVED_PROPOSALS == 200
    first = proposals._build_proposal(
        _create_request("first"), manifest.profile_id, scope.scope_id
    )
    second = proposals._build_proposal(
        _create_request("second"), manifest.profile_id, scope.scope_id
    )
    repository.commit_proposal(first, unresolved_limit=1)

    with pytest.raises(ProposalLimitExceededError):
        repository.commit_proposal(second, unresolved_limit=1)

    assert repository.list_proposals() == [first]


def test_accept_create_atomically_commits_record_manifest_outbox_and_receipt(
    tmp_path, memory_protector
) -> None:
    repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    proposal = proposals.create(
        _create_request(),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )

    accepted = proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    assert service.get_record(accepted.record_id) == accepted
    assert service.get_manifest().revision == manifest.revision + 1
    assert proposals.list_pending() == ()
    assert repository.list_proposals()[0].state is ProposalState.ACCEPTED
    assert accepted.provenance.source.value == "agent"
    assert accepted.provenance.actor is ActorType.USER
    assert accepted.provenance.reason_code == "user_approved_agent_proposal"
    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_outbox WHERE object_type = 'record'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_objects "
                "WHERE object_type = 'proposal' AND object_id = ?",
                (proposal.proposal_id,),
            ).fetchone()[0]
            == 1
        )


def test_update_proposal_inherits_controls_and_semantic_identity(
    tmp_path, memory_protector
) -> None:
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="long"
        ),
        semantic_key={"namespace": "preference", "subject": "stable-key"},
        controls=ProfileControls(
            sync_mode=SyncMode.DEVICE_ONLY,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
    )

    proposal = proposals.create(
        ProfileProposeRequest(
            operation="update",
            target_record_id=current.record_id,
            base_version_id=current.version_id,
            proposed_payload=PreferencePayload(
                subject="changed-by-agent", polarity="like", value="concise"
            ),
        ),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )

    assert proposal.proposed_record is not None
    assert proposal.proposed_record.controls == current.controls
    assert proposal.proposed_record.semantic_key == current.semantic_key
    accepted = proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)
    assert accepted.payload.value == "concise"


def test_workspace_promotion_accepts_as_new_global_record_with_provenance(
    tmp_path, memory_protector
) -> None:
    _repository, service, proposals, manifest, global_scope = _harness(
        tmp_path, memory_protector
    )
    workspace = service.create_workspace_scope("workspace-1", "Project")
    service.set_scope_authority(workspace.scope_id, "propose")
    source = service.create_manual_record(
        scope_id=workspace.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )

    proposal = proposals.create(
        ProfilePromoteRequest(
            source_record_id=source.record_id,
            base_version_id=source.version_id,
        ),
        profile_id=manifest.profile_id,
        scope_id=workspace.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    promoted = proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    assert proposal.operation.value == "promote"
    assert promoted.record_id != source.record_id
    assert promoted.scope_id == global_scope.scope_id
    assert promoted.provenance.derived_from_record_id == source.record_id
    assert service.get_record(source.record_id) == source


def test_working_context_proposal_defaults_to_thirty_day_expiry(
    tmp_path, memory_protector
) -> None:
    _repository, _service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )

    proposal = proposals.create(
        ProfileProposeRequest(
            operation="create",
            proposed_payload=WorkingContextPayload(
                subject="current task", value="ship it"
            ),
        ),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )

    assert proposal.proposed_record is not None
    assert proposal.proposed_record.expires_at == NOW + timedelta(days=30)


def test_accept_update_conflict_keeps_pending_proposal_and_current_record(
    tmp_path, memory_protector
) -> None:
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="long"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    proposal = proposals.create(
        ProfileProposeRequest(
            operation="update",
            target_record_id=current.record_id,
            base_version_id=current.version_id,
            proposed_payload=PreferencePayload(
                subject="response.detail", polarity="like", value="concise"
            ),
        ),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    newer = service.update_record(
        current.record_id,
        RecordMutation(
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value="medium"
            )
        ),
        expected_version_id=current.version_id,
    )

    with pytest.raises(ProfileConflictError):
        proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    assert service.get_record(current.record_id) == newer
    assert proposals.list_pending() == (proposal,)


def test_archive_proposal_accepts_as_new_archived_version(
    tmp_path, memory_protector
) -> None:
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    proposal = proposals.create(
        ProfileProposeRequest(
            operation="archive",
            target_record_id=current.record_id,
            base_version_id=current.version_id,
        ),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )

    archived = proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    assert archived.state.value == "archived"
    assert archived.parent_version_id == current.version_id
    assert archived.provenance.source.value == "agent"
    assert archived.provenance.actor is ActorType.USER


def test_accept_expired_proposal_shreds_content_without_applying_record(
    tmp_path, memory_protector
) -> None:
    now = [NOW]
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector, clock=lambda: now[0]
    )
    proposal = proposals.create(
        _create_request("EXPIRED-PROPOSAL-CANARY"),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    now[0] = NOW + timedelta(days=91)

    with pytest.raises(ValueError, match="proposal_expired"):
        proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    receipt = service._get_profile_proposal(proposal.proposal_id)
    assert receipt is not None
    assert receipt.state is ProposalState.EXPIRED
    assert receipt.proposed_record is None
    assert service.list_records(scope_ids=(scope.scope_id,)) == ()


def test_proposal_expiring_at_accept_transaction_is_shredded_not_rolled_back(
    tmp_path, memory_protector, monkeypatch
) -> None:
    now = [NOW]
    repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector, clock=lambda: now[0]
    )
    proposal = proposals.create(
        _create_request("ACCEPT-RACE-CANARY"),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    original_accept = service._accept_profile_proposal

    def advance_before_transaction(*args, **kwargs):
        now[0] = NOW + timedelta(days=91)
        return original_accept(*args, **kwargs)

    monkeypatch.setattr(service, "_accept_profile_proposal", advance_before_transaction)

    with pytest.raises(ValueError, match="proposal_expired"):
        proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    receipt = repository.get_proposal(proposal.proposal_id)
    assert receipt is not None and receipt.state is ProposalState.EXPIRED
    assert service.list_records(scope_ids=(scope.scope_id,)) == ()


def test_accept_time_collision_cannot_block_expired_proposal_shredding(
    tmp_path, memory_protector, monkeypatch
) -> None:
    now = [NOW]
    repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector, clock=lambda: now[0]
    )
    proposal = proposals.create(
        _create_request("COLLISION-RACE-CANARY"),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    collision = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="existing"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    original_accept = service._accept_profile_proposal

    def advance_before_transaction(*args, **kwargs):
        now[0] = NOW + timedelta(days=91)
        return original_accept(*args, **kwargs)

    monkeypatch.setattr(service, "_accept_profile_proposal", advance_before_transaction)

    with pytest.raises(ValueError, match="proposal_expired"):
        proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    receipt = repository.get_proposal(proposal.proposal_id)
    assert receipt is not None and receipt.state is ProposalState.EXPIRED
    assert service.get_record(collision.record_id) == collision


def test_global_scope_cannot_create_workspace_promotion_proposal(
    tmp_path, memory_protector
) -> None:
    _repository, service, proposals, manifest, global_scope = _harness(
        tmp_path, memory_protector
    )
    current = service.create_manual_record(
        scope_id=global_scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )

    with pytest.raises(PersonalContextAuthorityError) as caught:
        proposals.create(
            ProfilePromoteRequest(
                source_record_id=current.record_id,
                base_version_id=current.version_id,
            ),
            profile_id=manifest.profile_id,
            scope_id=global_scope.scope_id,
            turn_id="turn-1",
            session_id="session-1",
        )

    assert caught.value.reason_code == "promotion_requires_workspace"


@pytest.mark.parametrize("ineligible", ["archived", "expired", "private", "conflicted"])
def test_agent_update_rejects_every_ineligible_record_with_one_generic_reason(
    tmp_path, memory_protector, monkeypatch, ineligible: str
) -> None:
    now = [NOW]
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector, clock=lambda: now[0]
    )
    controls = {
        "sync_mode": "syncable",
        "agent_visibility": "user_only" if ineligible == "private" else "agent_visible",
    }
    if ineligible == "expired":
        payload = WorkingContextPayload(subject="current task", value="old")
        proposed_payload = WorkingContextPayload(subject="current task", value="new")
        current = service.create_manual_record(
            scope_id=scope.scope_id,
            payload=payload,
            semantic_key={"namespace": "working_context", "subject": "current task"},
            controls=controls,
            expires_at=NOW + timedelta(days=1),
        )
        now[0] = NOW + timedelta(days=2)
    else:
        proposed_payload = PreferencePayload(
            subject="response.detail", polarity="like", value="new"
        )
        current = service.create_manual_record(
            scope_id=scope.scope_id,
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value="old"
            ),
            semantic_key={"namespace": "preference", "subject": "response.detail"},
            controls=controls,
        )
    if ineligible == "archived":
        current = service.archive_record(
            current.record_id, expected_version_id=current.version_id
        )
    if ineligible == "conflicted":
        original_view = service.authorized_context_view

        def conflicted_view(**kwargs):
            return replace(
                original_view(**kwargs), conflicted_record_ids=(current.record_id,)
            )

        monkeypatch.setattr(service, "authorized_context_view", conflicted_view)

    with pytest.raises(PersonalContextAuthorityError) as caught:
        proposals.create(
            ProfileProposeRequest(
                operation="update",
                target_record_id=current.record_id,
                base_version_id=current.version_id,
                proposed_payload=proposed_payload,
            ),
            profile_id=manifest.profile_id,
            scope_id=scope.scope_id,
            turn_id="turn-1",
            session_id="session-1",
        )

    assert caught.value.reason_code == "record_ineligible"


@pytest.mark.parametrize("operation", ["archive", "promote", "direct_write"])
def test_every_agent_mutation_path_uses_the_shared_record_eligibility_guard(
    tmp_path, memory_protector, operation: str
) -> None:
    _repository, service, proposals, manifest, global_scope = _harness(
        tmp_path, memory_protector
    )
    scope = global_scope
    if operation == "promote":
        scope = service.create_workspace_scope("workspace-1", "Project")
        service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="private"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "user_only"},
    )
    if operation == "archive":
        request = ProfileProposeRequest(
            operation="archive",
            target_record_id=current.record_id,
            base_version_id=current.version_id,
        )
    elif operation == "promote":
        request = ProfilePromoteRequest(
            source_record_id=current.record_id,
            base_version_id=current.version_id,
        )
    else:
        service.set_scope_authority(scope.scope_id, AgentAuthority.DIRECT_WRITE)
        request = ProfileUpdateRequest(
            record_id=current.record_id,
            base_version_id=current.version_id,
            current_user_message_id="message-1",
            evidence_span="I prefer concise replies.",
            proposed_payload=PreferencePayload(
                subject="response.detail", polarity="like", value="concise"
            ),
        )

    with pytest.raises(PersonalContextAuthorityError) as caught:
        if operation == "direct_write":
            proposals.apply_direct_update(
                request,
                profile_id=manifest.profile_id,
                scope_id=scope.scope_id,
                evidence_hash="0" * 64,
            )
        else:
            proposals.create(
                request,
                profile_id=manifest.profile_id,
                scope_id=scope.scope_id,
                turn_id="turn-1",
                session_id="session-1",
            )

    assert caught.value.reason_code == "record_ineligible"


@pytest.mark.parametrize("changed_fence", ["global", "scope", "binding"])
def test_proposal_commit_fails_when_live_authority_fence_changes_before_transaction(
    tmp_path, memory_protector, monkeypatch, changed_fence: str
) -> None:
    _repository, service, proposals, manifest, _global_scope = _harness(
        tmp_path, memory_protector
    )
    scope = service.create_workspace_scope("workspace-1", "Project")
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    original_commit = service._commit_profile_proposal

    def raced_commit(*args, **kwargs):
        if changed_fence == "global":
            service.set_runtime_enabled(False)
        elif changed_fence == "scope":
            service.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)
        else:
            service.map_workspace_scope("workspace-2", scope.scope_id)
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(service, "_commit_profile_proposal", raced_commit)

    with pytest.raises(ProfileConflictError):
        proposals.create(
            _create_request(),
            profile_id=manifest.profile_id,
            scope_id=scope.scope_id,
            turn_id="turn-1",
            session_id="session-1",
        )

    assert service._list_profile_proposals() == ()


def test_direct_write_commit_fails_when_scope_authority_changes_before_transaction(
    tmp_path, memory_protector, monkeypatch
) -> None:
    _repository, service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    service.set_scope_authority(scope.scope_id, AgentAuthority.DIRECT_WRITE)
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="long"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    original_commit = service._commit_record

    def raced_commit(*args, **kwargs):
        service.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(service, "_commit_record", raced_commit)

    with pytest.raises(ProfileConflictError):
        proposals.apply_direct_update(
            ProfileUpdateRequest(
                record_id=current.record_id,
                base_version_id=current.version_id,
                current_user_message_id="message-1",
                evidence_span="I prefer concise replies.",
                proposed_payload=PreferencePayload(
                    subject="response.detail", polarity="like", value="concise"
                ),
            ),
            profile_id=manifest.profile_id,
            scope_id=scope.scope_id,
            evidence_hash="0" * 64,
        )

    assert service.get_record(current.record_id) == current


def test_promotion_acceptance_rechecks_source_head_inside_accept_transaction(
    tmp_path, memory_protector, monkeypatch
) -> None:
    _repository, service, proposals, manifest, global_scope = _harness(
        tmp_path, memory_protector
    )
    workspace = service.create_workspace_scope("workspace-1", "Project")
    service.set_scope_authority(workspace.scope_id, AgentAuthority.PROPOSE)
    source = service.create_manual_record(
        scope_id=workspace.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    proposal = proposals.create(
        ProfilePromoteRequest(
            source_record_id=source.record_id,
            base_version_id=source.version_id,
        ),
        profile_id=manifest.profile_id,
        scope_id=workspace.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    original_accept = service._accept_profile_proposal

    def raced_accept(*args, **kwargs):
        service.update_record(
            source.record_id,
            RecordMutation(
                payload=PreferencePayload(
                    subject="response.detail", polarity="like", value="changed"
                )
            ),
            expected_version_id=source.version_id,
        )
        return original_accept(*args, **kwargs)

    monkeypatch.setattr(service, "_accept_profile_proposal", raced_accept)

    with pytest.raises(ProfileConflictError):
        proposals.accept(proposal.proposal_id, user_actor=ActorType.USER)

    assert proposals.list_pending() == (proposal,)
    assert service.list_records(scope_ids=(global_scope.scope_id,)) == ()


def test_listing_pending_lazily_expires_and_shreds_due_proposals(
    tmp_path, memory_protector
) -> None:
    now = [NOW]
    repository, _service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector, clock=lambda: now[0]
    )
    proposal = proposals.create(
        _create_request("DUE-PROPOSAL-CANARY"),
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        turn_id="turn-1",
        session_id="session-1",
    )
    now[0] = NOW + timedelta(days=91)

    assert proposals.list_pending() == ()
    receipt = repository.get_proposal(proposal.proposal_id)
    assert receipt is not None and receipt.state is ProposalState.EXPIRED
    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_objects "
                "WHERE object_type = 'proposal' AND object_id = ?",
                (proposal.proposal_id,),
            ).fetchone()[0]
            == 1
        )


def test_repository_expires_due_proposals_before_enforcing_unresolved_ceiling(
    tmp_path, memory_protector
) -> None:
    repository, _service, proposals, manifest, scope = _harness(
        tmp_path, memory_protector
    )
    due = proposals._build_proposal(
        _create_request("due"), manifest.profile_id, scope.scope_id
    )
    replacement = proposals._build_proposal(
        _create_request("replacement"), manifest.profile_id, scope.scope_id
    )
    repository.commit_proposal(due, unresolved_limit=1)

    repository.commit_proposal(
        replacement,
        unresolved_limit=1,
        expire_before=NOW + timedelta(days=91),
    )

    states = {
        proposal.proposal_id: proposal.state for proposal in repository.list_proposals()
    }
    assert states == {
        due.proposal_id: ProposalState.EXPIRED,
        replacement.proposal_id: ProposalState.PENDING,
    }
