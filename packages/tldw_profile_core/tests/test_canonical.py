import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    RecordKind,
    RecordState,
    SemanticKey,
    SyncMode,
    canonical_bytes,
    integrity_tag,
    validate_profile_semantics,
)


def preference(
    record_id: str = "11111111-1111-4111-8111-111111111111",
) -> ProfileRecord:
    now = datetime(2026, 8, 28, tzinfo=UTC)
    return ProfileRecord(
        profile_id="22222222-2222-4222-8222-222222222222",
        record_id=record_id,
        scope_id="33333333-3333-4333-8333-333333333333",
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE, agent_visibility=AgentVisibility.AGENT_VISIBLE
        ),
        provenance=ProfileProvenance(
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id="44444444-4444-4444-8444-444444444444",
        parent_version_id=None,
        created_at=now,
        updated_at=now,
    )


def test_canonical_bytes_are_stable_and_whitespace_free():
    value = canonical_bytes(preference())
    assert value == canonical_bytes(ProfileRecord.model_validate_json(value))
    assert b"\n" not in value and b": " not in value


def test_jcs_bytes_and_hmac_match_cross_runtime_fixture():
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/v1/19-jcs-conformance.json").read_text(
            encoding="utf-8"
        )
    )
    proposal = ProfileProposal.model_validate(fixture["data"])
    expected = fixture["canonical_utf8"].encode("utf-8")
    assert canonical_bytes(proposal) == expected
    assert canonical_bytes(ProfileProposal.model_validate_json(expected)) == expected
    assert (
        integrity_tag(proposal, bytes.fromhex(fixture["canonical_key_hex"]))
        == fixture["integrity_tag"]
    )


def test_portable_timestamps_roundtrip_through_json_modes():
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/v1/19-jcs-conformance.json").read_text(
            encoding="utf-8"
        )
    )
    proposal = ProfileProposal.model_validate(fixture["data"])
    dumped = proposal.model_dump(mode="json")
    assert dumped["created_at"] == "2026-08-27T19:32:03.123Z"
    assert dumped["expires_at"] == "2026-11-25T19:32:03.123Z"
    assert ".123000Z" not in proposal.model_dump_json()
    assert ProfileProposal.model_validate(dumped) == proposal
    validate_profile_semantics(dumped)

    nested_data = preference().model_dump(mode="json")
    nested_data.update(
        profile_id=fixture["data"]["profile_id"],
        scope_id=fixture["data"]["scope_id"],
        created_at=fixture["data"]["created_at"],
        updated_at=fixture["data"]["created_at"],
    )
    nested_proposal = ProfileProposal.model_validate(
        {
            **fixture["data"],
            "operation": "create",
            "target_record_id": None,
            "base_version_id": None,
            "proposed_record": nested_data,
        }
    )
    nested_dump = nested_proposal.model_dump(mode="json")
    assert nested_dump["proposed_record"]["created_at"] == ("2026-08-27T19:32:03.123Z")
    assert ProfileProposal.model_validate(nested_dump) == nested_proposal
    validate_profile_semantics(nested_dump)


def test_integrity_tag_is_keyed_and_versioned():
    record = preference()
    assert integrity_tag(record, b"a" * 32).startswith("hmac-sha256-v1:")
    assert integrity_tag(record, b"a" * 32) != integrity_tag(record, b"b" * 32)


def test_same_scope_semantic_key_is_structured_not_free_text():
    assert preference().semantic_key == SemanticKey(
        namespace="preference", subject="response.detail"
    )


def test_integrity_key_must_be_32_bytes():
    with pytest.raises(ValueError):
        integrity_tag(preference(), b"short")
