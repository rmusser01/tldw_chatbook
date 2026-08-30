"""Governed durable Personal Context proposal operations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock
from typing import TYPE_CHECKING

from tldw_profile_core import (
    ActorType,
    AgentVisibility,
    ProfileControls,
    ProfilePayload,
    ProfilePromoteRequest,
    ProfileProposeRequest,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    ProposalOperation,
    ProposalState,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from .runtime_policy import AgentAuthority, PersonalContextAuthorityError

if TYPE_CHECKING:
    from .service import PersonalContextService


class ProposalQuotaExceeded(RuntimeError):
    """Report that a root turn or Console session exhausted proposal capacity."""


class PrivateDuplicateReviewRequired(RuntimeError):
    """Request generic user review without revealing a private duplicate."""


@dataclass(slots=True)
class _QuotaReservation:
    quota: "ProfileProposalQuota"
    turn_id: str
    session_id: str
    _released: bool = False

    def release(self) -> None:
        """Release one failed proposal commit reservation exactly once."""

        if not self._released:
            self.quota._release(self.turn_id, self.session_id)
            self._released = True


class ProfileProposalQuota:
    """Process-local proposal quota shared by fresh per-run providers."""

    def __init__(self, *, per_turn: int = 5, per_session: int = 25) -> None:
        self._per_turn = per_turn
        self._per_session = per_session
        self._turn_counts: dict[str, int] = defaultdict(int)
        self._session_counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def reserve(self, turn_id: str, session_id: str) -> _QuotaReservation:
        """Atomically reserve capacity for one proposal commit."""

        with self._lock:
            if (
                self._turn_counts[turn_id] >= self._per_turn
                or self._session_counts[session_id] >= self._per_session
            ):
                raise ProposalQuotaExceeded("proposal_quota_exceeded")
            self._turn_counts[turn_id] += 1
            self._session_counts[session_id] += 1
        return _QuotaReservation(self, turn_id, session_id)

    def _release(self, turn_id: str, session_id: str) -> None:
        with self._lock:
            self._turn_counts[turn_id] -= 1
            self._session_counts[session_id] -= 1


class ProfileProposalService:
    """Build and resolve canonical proposals through one app-owned service."""

    def __init__(
        self,
        service: "PersonalContextService",
        *,
        quota: ProfileProposalQuota,
    ) -> None:
        self._service = service
        self._quota = quota

    def create(
        self,
        request: ProfileProposeRequest | ProfilePromoteRequest,
        *,
        profile_id: str,
        scope_id: str,
        turn_id: str,
        session_id: str,
        evidence_reference: str | None = None,
        evidence_hash: str | None = None,
    ) -> ProfileProposal:
        """Create one pending canonical proposal without changing record context."""

        if isinstance(request, ProfilePromoteRequest):
            request = ProfilePromoteRequest.model_validate(request)
        else:
            request = ProfileProposeRequest.model_validate(request)
        authority_fence = self._service._capture_agent_authority_fence(
            scope_id, AgentAuthority.PROPOSE
        )
        manifest = self._service.get_manifest()
        if manifest.profile_id != profile_id:
            raise PermissionError("profile_scope_mismatch")
        scope = next(
            (
                candidate
                for candidate in self._service.list_scopes()
                if candidate.scope_id == scope_id
            ),
            None,
        )
        if scope is None or scope.profile_id != profile_id:
            raise PermissionError("profile_scope_mismatch")
        if (
            isinstance(request, ProfilePromoteRequest)
            and scope.kind is not ScopeKind.WORKSPACE
        ):
            raise PersonalContextAuthorityError("promotion_requires_workspace")
        self._require_no_private_duplicate(request, scope_id)
        reservation = self._quota.reserve(turn_id, session_id)
        try:
            proposal = self._build_proposal(
                request,
                profile_id,
                scope_id,
                evidence_reference=evidence_reference,
                evidence_hash=evidence_hash,
            )
            self._service._commit_profile_proposal(
                proposal, authority_fence=authority_fence
            )
        except BaseException:
            reservation.release()
            raise
        return proposal

    def _require_no_private_duplicate(
        self,
        request: ProfileProposeRequest | ProfilePromoteRequest,
        scope_id: str,
    ) -> None:
        if not isinstance(request, ProfileProposeRequest):
            return
        if (
            request.operation is not ProposalOperation.CREATE
            or request.proposed_payload is None
        ):
            return
        payload = request.proposed_payload
        semantic_key = SemanticKey(
            namespace=payload.kind,
            subject=getattr(payload, "subject", payload.kind),
        )
        for record in self._service.list_records(scope_ids=(scope_id,)):
            if (
                record.kind.value == payload.kind
                and record.semantic_key == semantic_key
                and record.controls.agent_visibility is AgentVisibility.USER_ONLY
            ):
                raise PrivateDuplicateReviewRequired("private_duplicate")

    def _build_proposal(
        self,
        request: ProfileProposeRequest | ProfilePromoteRequest,
        profile_id: str,
        scope_id: str,
        *,
        evidence_reference: str | None = None,
        evidence_hash: str | None = None,
    ) -> ProfileProposal:
        now = self._service.clock()
        source_references = (
            (evidence_reference,) if evidence_reference is not None else ()
        )
        source_hashes = (evidence_hash,) if evidence_hash is not None else ()
        provenance = ProfileProvenance(
            source="agent",
            actor="agent",
            reason_code="conversation_learning",
            source_references=source_references,
            source_hashes=source_hashes,
        )
        proposed_record = None
        if isinstance(request, ProfilePromoteRequest):
            source = self._require_current(
                request.source_record_id, request.base_version_id, scope_id
            )
            self._service._require_agent_eligible_record(source, scope_id)
            operation = ProposalOperation.PROMOTE
            target_record_id = source.record_id
            base_version_id = source.version_id
            confidence = None
        else:
            operation = request.operation
            target_record_id = request.target_record_id
            base_version_id = request.base_version_id
            confidence = request.confidence
        if operation is ProposalOperation.CREATE:
            assert request.proposed_payload is not None
            payload = request.proposed_payload
            subject = getattr(payload, "subject", payload.kind)
            proposed_record = ProfileRecord(
                profile_id=profile_id,
                record_id=self._service._new_profile_id("record"),
                scope_id=scope_id,
                kind=payload.kind,
                payload=payload,
                semantic_key=SemanticKey(namespace=payload.kind, subject=subject),
                state=RecordState.ACTIVE,
                controls=ProfileControls(
                    sync_mode=SyncMode.SYNCABLE,
                    agent_visibility=AgentVisibility.AGENT_VISIBLE,
                ),
                provenance=provenance,
                version_id=self._service._new_profile_id("record-version"),
                parent_version_id=None,
                created_at=now,
                updated_at=now,
                expires_at=(
                    now + timedelta(days=30)
                    if payload.kind == "working_context"
                    else None
                ),
            )
        elif operation is ProposalOperation.UPDATE:
            assert isinstance(request, ProfileProposeRequest)
            assert request.target_record_id is not None
            assert request.base_version_id is not None
            assert request.proposed_payload is not None
            current = self._require_current(
                request.target_record_id, request.base_version_id, scope_id
            )
            self._service._require_agent_eligible_record(current, scope_id)
            if request.proposed_payload.kind != current.kind.value:
                raise ValueError("Record kind cannot be changed.")
            proposed_record = ProfileRecord.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "payload": request.proposed_payload,
                    "semantic_key": current.semantic_key,
                    "controls": current.controls,
                    "provenance": provenance,
                    "version_id": self._service._new_profile_id("record-version"),
                    "parent_version_id": current.version_id,
                    "updated_at": now,
                }
            )
        elif operation is ProposalOperation.ARCHIVE:
            assert target_record_id is not None and base_version_id is not None
            current = self._require_current(target_record_id, base_version_id, scope_id)
            self._service._require_agent_eligible_record(current, scope_id)
        return ProfileProposal(
            proposal_id=self._service._new_profile_id("proposal"),
            profile_id=profile_id,
            scope_id=scope_id,
            operation=operation,
            target_record_id=target_record_id,
            base_version_id=base_version_id,
            proposed_record=proposed_record,
            provenance=provenance,
            confidence=confidence,
            created_at=now,
            expires_at=now + timedelta(days=90),
        )

    def _require_current(
        self, record_id: str, version_id: str, scope_id: str
    ) -> ProfileRecord:
        current = self._service.get_record(record_id)
        if (
            current is None
            or current.version_id != version_id
            or current.scope_id != scope_id
        ):
            raise ValueError("record_version_conflict")
        return current

    def accept(
        self,
        proposal_id: str,
        *,
        user_actor: ActorType,
        edited_payload: ProfilePayload | None = None,
    ) -> ProfileRecord:
        """Atomically apply one pending proposal and content-shred its content."""

        proposal = self._service._get_profile_proposal(proposal_id)
        if proposal is not None and proposal.state is ProposalState.EXPIRED:
            raise ValueError("proposal_expired")
        if proposal is None or proposal.state is not ProposalState.PENDING:
            raise ValueError("proposal_unavailable")
        if proposal.expires_at <= self._service.clock():
            self.expire(proposal_id)
            raise ValueError("proposal_expired")
        user_actor = ActorType(user_actor)
        if user_actor is not ActorType.USER:
            raise ValueError("proposal_acceptance_requires_user_actor")
        if edited_payload is not None and proposal.operation not in {
            ProposalOperation.CREATE,
            ProposalOperation.UPDATE,
        }:
            raise ValueError("proposal_operation_is_not_editable")
        expected_version = proposal.base_version_id
        allow_user_review_rewrite = False
        if proposal.operation in {ProposalOperation.CREATE, ProposalOperation.UPDATE}:
            assert proposal.proposed_record is not None
            proposed = proposal.proposed_record
            payload = edited_payload or proposed.payload
            assert payload is not None
            refresh_working_context = (
                proposed.kind.value == "working_context" and not proposed.no_expiry
            )
            if edited_payload is not None or refresh_working_context:
                if payload.kind != proposed.kind.value:
                    raise ValueError("proposal_kind_cannot_change")
                subject = getattr(payload, "subject", payload.kind)
                now = self._service.clock()
                proposed = ProfileRecord.model_validate(
                    {
                        **proposed.model_dump(mode="python"),
                        "payload": payload,
                        "semantic_key": SemanticKey(
                            namespace=payload.kind,
                            subject=subject,
                        ),
                        "version_id": self._service._new_profile_id("record-version"),
                        "updated_at": now,
                        "expires_at": (
                            now + timedelta(days=30)
                            if refresh_working_context
                            else proposed.expires_at
                        ),
                    }
                )
                allow_user_review_rewrite = True
            record = ProfileRecord.model_validate(
                {
                    **proposed.model_dump(mode="python"),
                    "provenance": self._approval_provenance(
                        proposal, user_actor=user_actor
                    ),
                }
            )
        elif proposal.operation is ProposalOperation.ARCHIVE:
            assert proposal.target_record_id is not None
            assert proposal.base_version_id is not None
            current = self._require_current(
                proposal.target_record_id,
                proposal.base_version_id,
                proposal.scope_id,
            )
            record = ProfileRecord.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "state": RecordState.ARCHIVED,
                    "provenance": self._approval_provenance(
                        proposal, user_actor=user_actor
                    ),
                    "version_id": self._service._new_profile_id("record-version"),
                    "parent_version_id": current.version_id,
                    "updated_at": self._service.clock(),
                }
            )
        else:
            assert proposal.target_record_id is not None
            assert proposal.base_version_id is not None
            source = self._require_current(
                proposal.target_record_id,
                proposal.base_version_id,
                proposal.scope_id,
            )
            global_scope = next(
                scope
                for scope in self._service.list_scopes()
                if scope.kind.value == "global"
            )
            now = self._service.clock()
            record = ProfileRecord.model_validate(
                {
                    **source.model_dump(mode="python"),
                    "record_id": self._service._new_profile_id("record"),
                    "scope_id": global_scope.scope_id,
                    "provenance": ProfileProvenance(
                        source="agent",
                        actor=user_actor,
                        reason_code="workspace_promotion",
                        derived_from_record_id=source.record_id,
                    ),
                    "version_id": self._service._new_profile_id("record-version"),
                    "parent_version_id": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            expected_version = None
        return self._service._accept_profile_proposal(
            proposal_id,
            record,
            expected_record_version=expected_version,
            allow_user_review_rewrite=allow_user_review_rewrite,
        )

    @staticmethod
    def _approval_provenance(
        proposal: ProfileProposal, *, user_actor: ActorType
    ) -> ProfileProvenance:
        return ProfileProvenance(
            source=proposal.provenance.source,
            actor=user_actor,
            reason_code="user_approved_agent_proposal",
            source_references=proposal.provenance.source_references,
            source_hashes=proposal.provenance.source_hashes,
            derived_from_record_id=proposal.provenance.derived_from_record_id,
        )

    def reject(self, proposal_id: str) -> ProfileProposal:
        return self._service._resolve_profile_proposal(
            proposal_id, ProposalState.REJECTED
        )

    def supersede(self, proposal_id: str) -> ProfileProposal:
        return self._service._resolve_profile_proposal(
            proposal_id, ProposalState.SUPERSEDED
        )

    def expire(self, proposal_id: str) -> ProfileProposal:
        return self._service._resolve_profile_proposal(
            proposal_id, ProposalState.EXPIRED
        )

    def apply_direct_update(
        self,
        request,
        *,
        profile_id: str,
        scope_id: str,
        evidence_hash: str,
    ) -> ProfileRecord:
        """Apply one exact-evidence direct update through the app service."""

        authority_fence = self._service._capture_agent_authority_fence(
            scope_id, AgentAuthority.DIRECT_WRITE
        )
        if self._service.get_manifest().profile_id != profile_id:
            raise PermissionError("profile_scope_mismatch")
        return self._service._apply_direct_profile_update(
            request,
            scope_id=scope_id,
            evidence_hash=evidence_hash,
            authority_fence=authority_fence,
        )

    def list_pending(self) -> tuple[ProfileProposal, ...]:
        """Return pending proposal heads for user-owned review surfaces."""

        return tuple(
            proposal
            for proposal in self._service._list_profile_proposals()
            if proposal.state is ProposalState.PENDING
        )


__all__ = [
    "ProfileProposalQuota",
    "ProfileProposalService",
    "PrivateDuplicateReviewRequired",
    "ProposalQuotaExceeded",
]
