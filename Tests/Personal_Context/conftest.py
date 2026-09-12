from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    ProposalOperation,
    ProvenanceSource,
    RecordKind,
    RecordState,
    SemanticKey,
    SyncMode,
)
from tldw_profile_core.models import ActorType

from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
)


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


@pytest.fixture
def memory_protector() -> InMemoryProfileKeyProtector:
    return InMemoryProfileKeyProtector()


@pytest.fixture
def record_factory():
    def make_record(
        profile_id: str,
        *,
        record_id: str = "record-1",
        version_id: str = "record-version-1",
        parent_version_id: str | None = None,
        value: str = "concise answers",
        sync_mode: SyncMode = SyncMode.SYNCABLE,
    ) -> ProfileRecord:
        return ProfileRecord(
            profile_id=profile_id,
            record_id=record_id,
            scope_id="scope-global",
            kind=RecordKind.PREFERENCE,
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value=value
            ),
            semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
            state=RecordState.ACTIVE,
            controls=ProfileControls(
                sync_mode=sync_mode,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            ),
            provenance=ProfileProvenance(
                source=ProvenanceSource.MANUAL,
                actor=ActorType.USER,
                reason_code="settings_edit",
            ),
            version_id=version_id,
            parent_version_id=parent_version_id,
            created_at=NOW,
            updated_at=NOW,
        )

    return make_record


@pytest.fixture
def proposal_factory(record_factory):
    def make_proposal(profile_id: str, *, proposal_id: str = "proposal-1"):
        return ProfileProposal(
            proposal_id=proposal_id,
            profile_id=profile_id,
            scope_id="scope-global",
            operation=ProposalOperation.CREATE,
            target_record_id=None,
            base_version_id=None,
            proposed_record=record_factory(
                profile_id,
                record_id="proposed-record-1",
                version_id="proposed-record-version-1",
            ),
            provenance=ProfileProvenance(
                source=ProvenanceSource.AGENT,
                actor=ActorType.AGENT,
                reason_code="conversation_learning",
            ),
            confidence=0.8,
            created_at=NOW,
            expires_at=NOW + timedelta(days=90),
        )

    return make_proposal
