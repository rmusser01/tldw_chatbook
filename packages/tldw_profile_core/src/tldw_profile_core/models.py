from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .enums import AgentVisibility, ProposalOperation, ProposalState, RecordKind, RecordState, ScopeKind, SyncMode
from .payloads import FactPayload, FrozenModel, GoalPayload, LegacyUnclassifiedPayload, PreferencePayload, ProfilePayload, WorkingContextPayload


class SemanticKey(FrozenModel):
    namespace: str
    subject: str


class ProfileControls(FrozenModel):
    sync_mode: SyncMode
    agent_visibility: AgentVisibility


class ProfileProvenance(FrozenModel):
    source: str
    actor: str
    reason_code: str


class ProfileManifest(FrozenModel):
    schema_version: Literal[1] = 1
    profile_id: str
    revision: int = Field(ge=0)
    purge_generation: int = Field(ge=0)


class ProfileScope(FrozenModel):
    schema_version: Literal[1] = 1
    scope_id: str
    profile_id: str
    kind: ScopeKind


class ProfileRecord(FrozenModel):
    schema_version: Literal[1] = 1
    profile_id: str
    record_id: str
    scope_id: str
    kind: RecordKind
    payload: ProfilePayload
    semantic_key: SemanticKey | None = None
    state: RecordState
    controls: ProfileControls
    provenance: ProfileProvenance
    version_id: str
    parent_version_id: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    no_expiry: bool = False

    @model_validator(mode="after")
    def validate_record(self):
        if self.kind.value != self.payload.kind:
            raise ValueError("record kind and payload kind must agree")
        if self.kind is RecordKind.WORKING_CONTEXT and self.expires_at is None and not self.no_expiry:
            raise ValueError("working context requires expires_at or no_expiry")
        if len(self.payload.model_dump_json()) > 16 * 1024:
            raise ValueError("payload exceeds 16 KiB")
        return self


class ProfileProposal(FrozenModel):
    schema_version: Literal[1] = 1
    proposal_id: str
    profile_id: str
    scope_id: str
    operation: ProposalOperation
    target_record_id: str | None
    base_version_id: str | None
    proposed_record: ProfileRecord | None
    provenance: ProfileProvenance
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    state: ProposalState = ProposalState.PENDING
    created_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validate_target_base(self):
        if self.operation in (ProposalOperation.UPDATE, ProposalOperation.ARCHIVE) and (not self.target_record_id or not self.base_version_id):
            raise ValueError("update/archive proposals require target and base")
        if self.operation is ProposalOperation.CREATE and self.proposed_record is None:
            raise ValueError("create proposals require proposed_record")
        return self


class ProfileSearchRequest(FrozenModel):
    profile_id: str
    scope_id: str | None = None
    query: str


class ProfileGetRequest(FrozenModel):
    profile_id: str
    record_id: str


class ProfileProposeRequest(ProfileProposal):
    pass


class ProfileUpdateRequest(FrozenModel):
    profile_id: str
    record_id: str
    base_version_id: str
    proposed_record: ProfileRecord


class ProfilePromoteRequest(FrozenModel):
    profile_id: str
    proposal_id: str
