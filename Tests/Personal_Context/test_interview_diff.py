from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tldw_profile_core import (
    AgentVisibility,
    ConstraintPayload,
    InterviewAudience,
    InterviewProposalBatch,
    InterviewProposedChange,
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    ProposalOperation,
    RecordState,
    SemanticKey,
    SyncMode,
    WorkingContextPayload,
)

from tldw_chatbook.Personal_Context.interview_diff import build_interview_diff


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
CONTROLS = ProfileControls(
    sync_mode=SyncMode.SYNCABLE,
    agent_visibility=AgentVisibility.AGENT_VISIBLE,
)


def _existing(*, visibility=AgentVisibility.AGENT_VISIBLE) -> ProfileRecord:
    return ProfileRecord(
        profile_id="profile-1",
        record_id="record-private"
        if visibility is AgentVisibility.USER_ONLY
        else "record-1",
        scope_id="scope-global",
        kind="preference",
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="detailed"
        ),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=visibility,
        ),
        provenance=ProfileProvenance(
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id="version-private"
        if visibility is AgentVisibility.USER_ONLY
        else "version-1",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _batch(*changes: InterviewProposedChange) -> InterviewProposalBatch:
    return InterviewProposalBatch(
        pack_id="pack-1",
        pack_version=1,
        audience=InterviewAudience.PERSONAL,
        changes=changes,
    )


def test_same_kind_semantic_key_becomes_deterministic_update_but_cross_kind_remains_addition() -> (
    None
):
    key = SemanticKey(namespace="preference", subject="response.detail")
    preference = InterviewProposedChange(
        operation=ProposalOperation.CREATE,
        proposed_payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        controls=CONTROLS,
        semantic_key=key,
    )
    constraint = InterviewProposedChange(
        operation=ProposalOperation.CREATE,
        proposed_payload=ConstraintPayload(
            subject="response.detail", value="never omit caveats"
        ),
        controls=CONTROLS,
        semantic_key=key,
    )

    first = build_interview_diff(
        _batch(preference, constraint), (_existing(),), now=NOW
    )
    second = build_interview_diff(
        _batch(constraint, preference), (_existing(),), now=NOW
    )

    assert {item.change.operation for item in first.changes} == {
        ProposalOperation.CREATE,
        ProposalOperation.UPDATE,
    }
    assert (
        next(
            item.change.target_record_id
            for item in first.changes
            if item.change.operation is ProposalOperation.UPDATE
        )
        == "record-1"
    )
    assert tuple(item.change_id for item in first.changes) == tuple(
        item.change_id for item in second.changes
    )


def test_user_only_record_is_never_exposed_and_only_sets_private_duplicate_indicator() -> (
    None
):
    canary = "PRIVATE_VALUE_CANARY_8b3f"
    private = _existing(visibility=AgentVisibility.USER_ONLY).model_copy(
        update={
            "payload": PreferencePayload(
                subject="response.detail", polarity="like", value=canary
            )
        }
    )
    proposed = InterviewProposedChange(
        operation=ProposalOperation.CREATE,
        proposed_payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        controls=CONTROLS,
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
    )

    result = build_interview_diff(_batch(proposed), (private,), now=NOW)

    assert result.changes[0].possible_private_duplicate is True
    assert result.changes[0].change.operation is ProposalOperation.CREATE
    assert canary not in repr(result)
    assert "record-private" not in repr(result)


def test_duplicate_proposed_changes_are_deduplicated_before_ids_become_ambiguous() -> (
    None
):
    proposed = InterviewProposedChange(
        operation=ProposalOperation.CREATE,
        proposed_payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        controls=CONTROLS,
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
    )

    result = build_interview_diff(_batch(proposed, proposed), (), now=NOW)

    assert len(result.changes) == 1


def test_repeated_semantic_topic_keeps_last_exact_answer_without_text_merge() -> None:
    first = InterviewProposedChange(
        operation=ProposalOperation.CREATE,
        proposed_payload=PreferencePayload(
            subject="response.detail", polarity="like", value="detailed"
        ),
        controls=CONTROLS,
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
    )
    last = first.model_copy(
        update={
            "proposed_payload": PreferencePayload(
                subject="response.detail", polarity="like", value="concise"
            )
        }
    )

    result = build_interview_diff(_batch(first, last), (), now=NOW)

    assert len(result.changes) == 1
    assert result.changes[0].change.proposed_payload.value == "concise"


def test_expired_active_record_is_not_an_update_target() -> None:
    expired = ProfileRecord(
        profile_id="profile-1",
        record_id="record-expired",
        scope_id="scope-workspace",
        kind="working_context",
        payload=WorkingContextPayload(
            subject="working_context", value="obsolete context"
        ),
        semantic_key=SemanticKey(
            namespace="working_context", subject="working_context"
        ),
        state=RecordState.ACTIVE,
        controls=CONTROLS,
        provenance=ProfileProvenance(
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id="version-expired",
        parent_version_id=None,
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(days=2),
        expires_at=NOW - timedelta(days=1),
    )
    proposed = InterviewProposedChange(
        operation=ProposalOperation.CREATE,
        proposed_payload=WorkingContextPayload(
            subject="working_context", value="fresh context"
        ),
        controls=CONTROLS,
        semantic_key=SemanticKey(
            namespace="working_context", subject="working_context"
        ),
    )

    result = build_interview_diff(_batch(proposed), (expired,), now=NOW)

    assert result.changes[0].change.operation is ProposalOperation.CREATE
    assert result.changes[0].change.target_record_id is None
