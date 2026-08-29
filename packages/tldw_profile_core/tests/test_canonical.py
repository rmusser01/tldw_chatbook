from datetime import UTC, datetime

import pytest

from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    RecordKind,
    RecordState,
    SemanticKey,
    SyncMode,
    canonical_bytes,
    integrity_tag,
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
