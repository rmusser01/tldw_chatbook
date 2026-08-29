from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tldw_profile_core import (
    ActorType,
    AgentVisibility,
    ConstraintPayload,
    ConventionPayload,
    CorrectionPayload,
    GoalPayload,
    IdentityPayload,
    LegacyUnclassifiedPayload,
    PreferencePayload,
    ProfileControls,
    ProfileGetRequest,
    ProfileManifest,
    ProfilePromoteRequest,
    ProfileProposal,
    ProfileProposeRequest,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    ProfileSearchRequest,
    ProfileToolResult,
    ProfileUpdateRequest,
    ProposalOperation,
    ProposalState,
    ProvenanceSource,
    RecordKind,
    RecordState,
    RelationshipPayload,
    ScopeKind,
    SemanticKey,
    SyncMode,
    ToolOperation,
    ToolResultStatus,
    WorkingContextPayload,
)

NOW = datetime(2026, 8, 28, tzinfo=UTC)
PROFILE_ID, SCOPE_ID, RECORD_ID, VERSION_ID = (
    "profile-migrated-1",
    "scope-global",
    "record-legacy-7",
    "version-4",
)


def provenance(**changes):
    values = {"source": "manual", "actor": "user", "reason_code": "settings_edit"}
    values.update(changes)
    return ProfileProvenance(**values)


def record(**changes):
    values = dict(
        profile_id=PROFILE_ID,
        record_id=RECORD_ID,
        scope_id=SCOPE_ID,
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE, agent_visibility=AgentVisibility.AGENT_VISIBLE
        ),
        provenance=provenance(),
        version_id=VERSION_ID,
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )
    values.update(changes)
    return ProfileRecord(**values)


def update_record(**changes):
    values = {"version_id": "version-5", "parent_version_id": VERSION_ID}
    values.update(changes)
    return record(**values)


def proposal(operation, **changes):
    shapes = {
        ProposalOperation.CREATE: dict(
            target_record_id=None,
            base_version_id=None,
            proposed_record=record(record_id="new-record", version_id="new-version"),
        ),
        ProposalOperation.UPDATE: dict(
            target_record_id=RECORD_ID,
            base_version_id=VERSION_ID,
            proposed_record=update_record(),
        ),
        ProposalOperation.ARCHIVE: dict(
            target_record_id=RECORD_ID, base_version_id=VERSION_ID, proposed_record=None
        ),
        ProposalOperation.PROMOTE: dict(
            target_record_id=RECORD_ID, base_version_id=VERSION_ID, proposed_record=None
        ),
    }
    values = dict(
        proposal_id="proposal-1",
        profile_id=PROFILE_ID,
        scope_id=SCOPE_ID,
        operation=operation,
        provenance=provenance(source="agent", actor="agent", reason_code="suggestion"),
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
        **shapes[operation],
    )
    values.update(changes)
    return ProfileProposal(**values)


def test_v1_enum_vocabularies_are_exact():
    assert {x.value for x in RecordKind} == {
        "identity",
        "preference",
        "relationship",
        "correction",
        "constraint",
        "goal",
        "convention",
        "working_context",
        "legacy_unclassified",
    }
    assert {x.value for x in RecordState} == {"active", "archived", "deleted"}
    assert {x.value for x in ProposalOperation} == {
        "create",
        "update",
        "archive",
        "promote",
    }
    assert {x.value for x in ProposalState} == {
        "pending",
        "accepted",
        "rejected",
        "superseded",
        "expired",
    }
    assert {x.value for x in ToolOperation} == {
        "search",
        "get",
        "propose",
        "update",
        "promote",
    }
    assert {x.value for x in ToolResultStatus} == {
        "applied",
        "proposal_created",
        "review_required",
        "permission_denied",
        "quota_exceeded",
        "conflict",
        "profile_locked",
    }


@pytest.mark.parametrize(
    ("payload_type", "kwargs"),
    [
        (IdentityPayload, {"subject": "name", "value": "Ada"}),
        (
            PreferencePayload,
            {"subject": "format", "polarity": "like", "value": "brief"},
        ),
        (RelationshipPayload, {"subject": "Alex", "value": "colleague"}),
        (CorrectionPayload, {"subject": "name", "value": "Ada, not Ava"}),
        (ConstraintPayload, {"subject": "schedule", "value": "no mornings"}),
        (GoalPayload, {"subject": "release", "outcome": "ship safely"}),
        (ConventionPayload, {"subject": "commits", "value": "imperative"}),
        (WorkingContextPayload, {"subject": "task", "value": "profile core"}),
        (LegacyUnclassifiedPayload, {"text": "migrated text"}),
    ],
)
def test_every_payload_is_versioned_bounded_non_empty_and_frozen(payload_type, kwargs):
    payload = payload_type(**kwargs)
    field = next(iter(kwargs))
    assert payload.schema_version == 1
    with pytest.raises(ValidationError):
        payload_type(**{**kwargs, field: "   "})
    with pytest.raises(ValidationError):
        payload_type(**{**kwargs, field: "x" * 16385})
    with pytest.raises(ValidationError):
        setattr(payload, field, "changed")


@pytest.mark.parametrize("bad_id", ["", "   ", "x" * 129])
def test_opaque_ids_reject_blank_or_overlong_values(bad_id):
    with pytest.raises(ValidationError):
        ProfileGetRequest(record_id=bad_id)


def test_manifest_and_scope_require_aware_ordered_version_metadata():
    manifest = ProfileManifest(
        profile_id=PROFILE_ID,
        revision=2,
        purge_generation=1,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=1),
        current_version_id=VERSION_ID,
    )
    scope = ProfileScope(
        profile_id=PROFILE_ID,
        scope_id=SCOPE_ID,
        kind=ScopeKind.GLOBAL,
        version_id=VERSION_ID,
        created_at=NOW,
        updated_at=NOW,
    )
    with pytest.raises(ValidationError):
        ProfileManifest(
            **{**manifest.model_dump(), "updated_at": NOW - timedelta(seconds=1)}
        )
    with pytest.raises(ValidationError):
        ProfileScope(**{**scope.model_dump(), "created_at": NOW.replace(tzinfo=None)})
    with pytest.raises(ValidationError):
        ProfileScope(
            **scope.model_dump(), label="plaintext", workspace_id="app-workspace"
        )


@pytest.mark.parametrize("field", ["revision", "purge_generation"])
def test_manifest_integer_counters_stay_in_i_json_exact_range(field):
    values = {
        "profile_id": PROFILE_ID,
        "revision": 0,
        "purge_generation": 0,
        "created_at": NOW,
        "updated_at": NOW,
        "current_version_id": VERSION_ID,
    }
    values[field] = 2**53
    with pytest.raises(ValidationError):
        ProfileManifest(**values)


def test_timestamps_reject_submillisecond_precision():
    with pytest.raises(ValidationError):
        ProfileManifest(
            profile_id=PROFILE_ID,
            revision=0,
            purge_generation=0,
            created_at=NOW.replace(microsecond=123_456),
            updated_at=NOW.replace(microsecond=123_456),
            current_version_id=VERSION_ID,
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        1_787_875_200,
        "2026-08-28 00:00:00Z",
        "2026-08-28t00:00:00z",
        "2026-08-28T00:00:00.1230Z",
    ],
)
def test_manifest_rejects_nonportable_wire_timestamps(timestamp):
    with pytest.raises(ValidationError):
        ProfileManifest(
            profile_id=PROFILE_ID,
            revision=0,
            purge_generation=0,
            created_at=timestamp,
            updated_at=timestamp,
            current_version_id=VERSION_ID,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "2"),
        ("revision", True),
        ("purge_generation", "2"),
        ("purge_generation", True),
    ],
)
def test_manifest_rejects_coerced_integer_counters(field, value):
    values = {
        "profile_id": PROFILE_ID,
        "revision": 0,
        "purge_generation": 0,
        "created_at": NOW,
        "updated_at": NOW,
        "current_version_id": VERSION_ID,
    }
    values[field] = value
    with pytest.raises(ValidationError):
        ProfileManifest(**values)


def test_provenance_is_bounded_typed_and_immutable():
    value = provenance(
        source=ProvenanceSource.IMPORT,
        actor=ActorType.SYSTEM,
        source_references=("message-1",),
        source_hashes=("a" * 64,),
        derived_from_record_id="record-old",
    )
    assert isinstance(value.source_references, tuple)
    with pytest.raises(ValidationError):
        provenance(reason_code=" ")
    with pytest.raises(ValidationError):
        provenance(source_hashes=("not-sha256",))
    with pytest.raises(ValidationError):
        provenance(source_references=tuple(f"ref-{i}" for i in range(33)))


def test_deleted_tombstones_are_content_free_and_active_records_require_payloads():
    assert record(state="deleted", payload=None, semantic_key=None).payload is None
    for forbidden in (
        {"payload": record().payload},
        {"semantic_key": record().semantic_key},
        {"no_expiry": True},
    ):
        values = {
            "state": "deleted",
            "payload": None,
            "semantic_key": None,
            **forbidden,
        }
        with pytest.raises(ValidationError):
            record(**values)
    with pytest.raises(ValidationError):
        record(payload=None)
    with pytest.raises(ValidationError):
        record(kind="goal", payload=record().payload)


def test_only_working_context_has_exactly_one_expiry_decision():
    context = WorkingContextPayload(subject="task", value="ship")
    with pytest.raises(ValidationError):
        record(kind="working_context", payload=context)
    assert record(kind="working_context", payload=context, no_expiry=True).no_expiry
    assert record(
        kind="working_context", payload=context, expires_at=NOW + timedelta(hours=1)
    ).expires_at
    with pytest.raises(ValidationError):
        record(
            kind="working_context",
            payload=context,
            expires_at=NOW + timedelta(hours=1),
            no_expiry=True,
        )
    with pytest.raises(ValidationError):
        record(expires_at=NOW + timedelta(hours=1))


def test_payload_limit_counts_canonical_utf8_bytes_not_characters():
    with pytest.raises(ValidationError, match="16 KiB"):
        record(
            payload=PreferencePayload(subject="x", polarity="like", value="界" * 9000)
        )


@pytest.mark.parametrize("operation", list(ProposalOperation))
def test_proposal_operation_shapes_accept_only_their_exact_fields(operation):
    assert proposal(operation).operation is operation
    bad = {
        ProposalOperation.CREATE: {"target_record_id": RECORD_ID},
        ProposalOperation.UPDATE: {"base_version_id": None},
        ProposalOperation.ARCHIVE: {"proposed_record": update_record()},
        ProposalOperation.PROMOTE: {"proposed_record": update_record()},
    }[operation]
    with pytest.raises(ValidationError):
        proposal(operation, **bad)


@pytest.mark.parametrize(
    "change",
    [
        {"profile_id": "other-profile"},
        {"scope_id": "other-scope"},
        {"record_id": "other-record"},
        {"parent_version_id": "other-version"},
    ],
)
def test_update_proposal_validates_record_and_base_consistency(change):
    with pytest.raises(ValidationError, match="identity|base"):
        proposal("update", proposed_record=update_record(**change))


def test_proposal_timestamps_are_aware_and_ordered():
    with pytest.raises(ValidationError):
        proposal("archive", created_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        proposal("archive", expires_at=NOW)


def test_resolved_proposals_are_content_free_receipts():
    assert (
        proposal(
            "create", state=ProposalState.ACCEPTED, proposed_record=None
        ).proposed_record
        is None
    )
    assert (
        proposal(
            "update", state=ProposalState.ACCEPTED, proposed_record=None
        ).target_record_id
        == RECORD_ID
    )
    assert proposal("archive", state=ProposalState.ACCEPTED).proposed_record is None


def test_resolved_proposals_reject_retained_content_or_confidence():
    with pytest.raises(ValidationError):
        proposal("archive", state=ProposalState.ACCEPTED, confidence=0.5)
    with pytest.raises(ValidationError):
        proposal(
            "archive", state=ProposalState.ACCEPTED, proposed_record=update_record()
        )


def test_pending_proposal_expiry_is_limited_to_ninety_days():
    with pytest.raises(ValidationError):
        proposal("archive", expires_at=NOW + timedelta(days=91))


def test_pending_create_requires_content():
    with pytest.raises(ValidationError):
        proposal("create", proposed_record=None)


@pytest.mark.parametrize(
    "operation", [ProposalOperation.CREATE, ProposalOperation.UPDATE]
)
@pytest.mark.parametrize("state", [RecordState.ARCHIVED, RecordState.DELETED])
def test_pending_create_and_update_require_active_proposed_records(operation, state):
    proposed = (
        record(state=state, payload=None, semantic_key=None)
        if state is RecordState.DELETED
        else record(state=state)
    )
    if operation is ProposalOperation.UPDATE:
        proposed = proposed.model_copy(
            update={"version_id": "version-5", "parent_version_id": VERSION_ID}
        )
    with pytest.raises(ValidationError):
        proposal(operation, proposed_record=proposed)


def test_pending_expiry_is_exactly_ninety_days():
    assert proposal(
        "archive", expires_at=NOW + timedelta(days=90)
    ).expires_at == NOW + timedelta(days=90)
    for days in (1, 91):
        with pytest.raises(ValidationError):
            proposal("archive", expires_at=NOW + timedelta(days=days))


def test_inferred_proposal_can_omit_evidence_and_supply_confidence():
    request = ProfileProposeRequest(
        operation=ProposalOperation.CREATE,
        proposed_payload={
            "kind": "preference",
            "subject": "format",
            "polarity": "like",
            "value": "brief",
        },
        confidence=0.75,
    )
    assert request.evidence_span is None
    with pytest.raises(ValidationError):
        ProfileProposeRequest(
            operation=ProposalOperation.CREATE,
            proposed_payload=request.proposed_payload,
            confidence=1.1,
        )


@pytest.mark.parametrize(
    "secret",
    [
        "-----BEGIN PRIVATE KEY-----",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "password: hunter2-secret",
        "API key: api-example-secret",
        "access token: token-example-secret",
        "token: abcdefghijklmnopqrstuvwxyz",
        "token: 123456-abcdef",
        "credential: hunter2-secret",
    ],
)
def test_agent_payload_and_evidence_boundaries_reject_secret_material(secret):
    sensitive_payload = PreferencePayload(
        subject="credential", polarity="like", value=secret
    )
    safe_payload = PreferencePayload(subject="format", polarity="like", value="brief")
    with pytest.raises(ValidationError):
        ProfileProposeRequest(operation="create", proposed_payload=sensitive_payload)
    with pytest.raises(ValidationError):
        ProfileProposeRequest(
            operation="create", proposed_payload=safe_payload, evidence_span=secret
        )
    with pytest.raises(ValidationError):
        ProfileUpdateRequest(
            record_id=RECORD_ID,
            base_version_id=VERSION_ID,
            current_user_message_id="message-1",
            evidence_span="I prefer brief answers",
            proposed_payload=sensitive_payload,
        )
    with pytest.raises(ValidationError):
        ProfileUpdateRequest(
            record_id=RECORD_ID,
            base_version_id=VERSION_ID,
            current_user_message_id="message-1",
            evidence_span=secret,
            proposed_payload=safe_payload,
        )


def test_manual_profile_records_may_contain_sensitive_data():
    value = record(
        payload=PreferencePayload(
            subject="credential", polarity="like", value="password: hunter2-secret"
        )
    )
    assert value.payload.value == "password: hunter2-secret"


@pytest.mark.parametrize(
    "benign",
    [
        "The token is limited to 4096 characters",
        "token: 128000",
        "token = 128000",
    ],
)
def test_agent_boundaries_allow_benign_token_limit_wording(benign):
    payload = PreferencePayload(subject="format", polarity="like", value=benign)
    assert (
        ProfileProposeRequest(
            operation="create", proposed_payload=payload, evidence_span=benign
        ).evidence_span
        == benign
    )
    assert (
        ProfileUpdateRequest(
            record_id=RECORD_ID,
            base_version_id=VERSION_ID,
            current_user_message_id="message-1",
            evidence_span=benign,
            proposed_payload=payload,
        ).evidence_span
        == benign
    )


def test_search_and_get_do_not_accept_profile_or_scope_selection():
    assert ProfileSearchRequest(query="response").limit == 5
    assert ProfileSearchRequest(query="response", limit=20).limit == 20
    with pytest.raises(ValidationError):
        ProfileSearchRequest(query="response", limit=21)
    for forbidden in (
        {"profile_id": PROFILE_ID},
        {"scope_id": SCOPE_ID},
        {"workspace_id": "workspace"},
    ):
        with pytest.raises(ValidationError):
            ProfileSearchRequest(query="response", **forbidden)
        with pytest.raises(ValidationError):
            ProfileGetRequest(record_id=RECORD_ID, **forbidden)


def test_agent_propose_supports_only_pending_create_update_archive_shapes():
    ProfileProposeRequest(
        operation="create",
        proposed_payload=record().payload,
        evidence_span="I like brief answers",
    )
    ProfileProposeRequest(
        operation="update",
        target_record_id=RECORD_ID,
        base_version_id=VERSION_ID,
        proposed_payload=record().payload,
        evidence_span="Use details now",
    )
    ProfileProposeRequest(
        operation="archive",
        target_record_id=RECORD_ID,
        base_version_id=VERSION_ID,
        evidence_span="No longer applies",
    )
    with pytest.raises(ValidationError):
        ProfileProposeRequest(
            operation="archive",
            evidence_span="archive",
            proposed_payload=record().payload,
        )


@pytest.mark.parametrize(
    ("request_type", "valid", "forbidden"),
    [
        (
            ProfileProposeRequest,
            {
                "operation": "create",
                "proposed_payload": {
                    "kind": "preference",
                    "subject": "x",
                    "polarity": "like",
                    "value": "y",
                },
                "evidence_span": "I like y",
            },
            [
                "controls",
                "state",
                "profile_id",
                "scope_id",
                "workspace_id",
                "proposal_id",
                "purge",
                "delete",
                "promote",
            ],
        ),
        (
            ProfileUpdateRequest,
            {
                "record_id": RECORD_ID,
                "base_version_id": VERSION_ID,
                "current_user_message_id": "message-1",
                "evidence_span": "I prefer y",
                "proposed_payload": {
                    "kind": "preference",
                    "subject": "x",
                    "polarity": "like",
                    "value": "y",
                },
            },
            [
                "controls",
                "state",
                "profile_id",
                "scope_id",
                "workspace_id",
                "proposal_id",
                "purge",
                "delete",
            ],
        ),
        (
            ProfilePromoteRequest,
            {"source_record_id": RECORD_ID, "base_version_id": VERSION_ID},
            [
                "profile_id",
                "scope_id",
                "workspace_id",
                "proposal_id",
                "controls",
                "state",
                "proposed_payload",
            ],
        ),
    ],
)
def test_agent_mutation_requests_reject_authority_privacy_delete_purge_and_accepted_state(
    request_type, valid, forbidden
):
    request_type(**valid)
    for field in forbidden:
        with pytest.raises(ValidationError):
            request_type(**valid, **{field: "forbidden"})
    with pytest.raises(ValidationError):
        request_type(**valid, proposal_state="accepted")


def test_update_evidence_is_non_empty_and_bounded():
    values = dict(
        record_id=RECORD_ID,
        base_version_id=VERSION_ID,
        current_user_message_id="message-1",
        proposed_payload=record().payload,
    )
    with pytest.raises(ValidationError):
        ProfileUpdateRequest(**values, evidence_span="   ")
    with pytest.raises(ValidationError):
        ProfileUpdateRequest(**values, evidence_span="x" * 1001)


def test_tool_result_has_only_explicit_operation_and_status_vocabularies():
    assert (
        ProfileToolResult(operation="search", status="applied", message="ok").status
        is ToolResultStatus.APPLIED
    )
    with pytest.raises(ValidationError):
        ProfileToolResult(operation="delete", status="applied", message="no")
    with pytest.raises(ValidationError):
        ProfileToolResult(operation="search", status="accepted", message="no")
