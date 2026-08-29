from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tldw_profile_core import (
    AgentVisibility,
    InterviewPack,
    InterviewQuestion,
    PreferencePayload,
    ProfileControls,
    ProfileProposeRequest,
    ProfileProvenance,
    ProfileRecord,
    ProposalOperation,
    RecordKind,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
    ToolOperation,
    ProfileToolResult,
)


NOW = datetime(2026, 8, 28, tzinfo=UTC)


def record(**changes):
    values = dict(
        profile_id="22222222-2222-4222-8222-222222222222",
        record_id="11111111-1111-4111-8111-111111111111",
        scope_id="33333333-3333-4333-8333-333333333333",
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(subject="response.detail", polarity="like", value="concise"),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(sync_mode=SyncMode.SYNCABLE, agent_visibility=AgentVisibility.AGENT_VISIBLE),
        provenance=ProfileProvenance(source="manual", actor="user", reason_code="settings_edit"),
        version_id="44444444-4444-4444-8444-444444444444",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )
    values.update(changes)
    if values["kind"] is RecordKind.WORKING_CONTEXT:
        values["payload"] = {"kind": "working_context", "subject": "task", "value": "ship"}
    return ProfileRecord(**values)


def test_models_forbid_unknown_fields_and_record_confidence():
    with pytest.raises(ValidationError):
        record(unexpected=True)
    with pytest.raises(ValidationError):
        record(confidence=0.5)


def test_working_context_requires_expiry_or_explicit_no_expiry():
    with pytest.raises(ValidationError):
        record(kind=RecordKind.WORKING_CONTEXT, expires_at=None, no_expiry=False)
    assert record(kind=RecordKind.WORKING_CONTEXT, no_expiry=True).no_expiry
    assert record(kind=RecordKind.WORKING_CONTEXT, expires_at=NOW + timedelta(hours=1)).expires_at


def test_kind_and_payload_must_agree():
    with pytest.raises(ValidationError):
        record(kind=RecordKind.PREFERENCE, payload={"subject": "x", "value": "y"})


def test_payload_is_limited_to_16_kibibytes_in_canonical_form():
    with pytest.raises(ValidationError):
        record(payload=PreferencePayload(subject="x", polarity="like", value="a" * 17000))


def test_proposal_update_requires_target_and_base():
    with pytest.raises(ValidationError):
        ProfileProposeRequest(
            proposal_id="p", profile_id="p", scope_id="s", operation=ProposalOperation.UPDATE,
            target_record_id=None, base_version_id=None, proposed_record=None,
            provenance=ProfileProvenance(source="agent", actor="agent", reason_code="suggestion"),
            created_at=NOW, expires_at=NOW + timedelta(days=1),
        )


def test_interview_pack_rejects_compound_or_more_than_twenty_questions():
    with pytest.raises(ValidationError):
        InterviewPack(pack_id="p", questions=[InterviewQuestion(question_id="q", text="What and why?")])
    questions = [InterviewQuestion(question_id=str(i), text=f"Question {i}?") for i in range(21)]
    with pytest.raises(ValidationError):
        InterviewPack(pack_id="p", questions=questions)


@pytest.mark.parametrize("operation", [ToolOperation.DELETE, ToolOperation.PURGE, ToolOperation.PRIVACY_CONTROL, ToolOperation.CROSS_WORKSPACE])
def test_tool_contract_rejects_privacy_delete_purge_and_cross_workspace(operation):
    with pytest.raises(ValidationError):
        ProfileToolResult(operation=operation, ok=True, message="no")
